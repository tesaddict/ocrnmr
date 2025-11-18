"""Display module for OCR test progress using prompt_toolkit."""

import os
import threading
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from collections import deque

from prompt_toolkit import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, HSplit, VSplit, Window, FormattedTextControl
from prompt_toolkit.layout.dimension import D
from prompt_toolkit.styles import Style
from typing import Any

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
        
        # Configuration display
        self.config_info: Optional[str] = None
        
        # Rename preview
        self.rename_preview: List[Tuple[str, Optional[str], Optional[str]]] = []
        self.show_name = ""
        self.season = 0
        self.matched_count = 0
        self.total_count = 0
        self.show_preview = False
        self.preview_layout: Optional[Layout] = None
        
        # UI components
        self.app: Optional[Application] = None
        self.running = False
        self.early_exit_requested = False
        
        # Terminal dimensions (calculated once at startup)
        terminal_width, terminal_height = self._get_terminal_size()
        
        # Ensure minimum terminal size - if too small, use minimums
        MIN_WIDTH = 80
        MIN_HEIGHT = 24
        terminal_width = max(MIN_WIDTH, terminal_width)
        terminal_height = max(MIN_HEIGHT, terminal_height)
        
        self.terminal_width = terminal_width
        self.terminal_height = terminal_height
        
        # Derived dimensions (calculated once at startup)
        # Ensure dimensions don't exceed terminal size to prevent "window too small" errors
        # Top and bottom sections split terminal height 50/50
        self.top_section_height = terminal_height // 2
        self.bottom_section_height = terminal_height - self.top_section_height
        
        # Ensure both sections have minimum size
        MIN_SECTION_HEIGHT = 10
        if self.top_section_height < MIN_SECTION_HEIGHT:
            self.top_section_height = MIN_SECTION_HEIGHT
            self.bottom_section_height = terminal_height - self.top_section_height
        if self.bottom_section_height < MIN_SECTION_HEIGHT:
            self.bottom_section_height = MIN_SECTION_HEIGHT
            self.top_section_height = terminal_height - self.bottom_section_height
        
        # Pane width: split terminal width equally, accounting for splitter
        # Ensure total width doesn't exceed terminal width
        calculated_pane_width = (terminal_width - 1) // 2  # Account for splitter (1 char)
        self.pane_width = max(30, calculated_pane_width)  # Minimum 30 chars per pane
        
        # Verify total width doesn't exceed terminal
        total_width_needed = (self.pane_width * 2) + 1  # 2 panes + 1 splitter
        if total_width_needed > terminal_width:
            # Adjust pane width to fit
            self.pane_width = (terminal_width - 1) // 2
        
        # Activity log max lines (calculated once)
        fixed_lines = 7  # Header, quit message, blank, file info, blank, status, separator
        self.activity_log_max_lines = max(3, self.top_section_height - fixed_lines - 2)  # Minimum 3 lines
        
    def _get_terminal_size(self) -> Tuple[int, int]:
        """Get terminal dimensions (only called once during initialization)."""
        try:
            import shutil
            size = shutil.get_terminal_size()
            return (size.columns, size.lines)
        except Exception:
            # Defaults if we can't determine terminal size
            return (80, 50)
    
    def _get_ocr_pane_text(self):
        """Get text for OCR pane."""
        with self.lock:
            lines = []
            lines.append(("class:header", " OCR Processing "))
            lines.append(("class:log", " (Press 'e' for early exit, 'q' to quit, Ctrl+C to cancel)\n"))
            lines.append(("", "\n"))
            
            # Show current processing status
            if self.current_file:
                lines.append(("class:label", f"Processing File "))
                lines.append(("class:value", f"{self.current_file_index + 1}/{self.total_files}: {Path(self.current_file).name}\n"))
            else:
                lines.append(("class:label", f"Processing File "))
                lines.append(("class:value", f"-/{self.total_files}: -\n"))
            
            lines.append(("", "\n"))
            lines.append(("class:label", "Status: "))
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
            lines.append(("class:value", f"{status_display}\n"))
            
            # Show activity log - newest at top (bottom-up scrolling)
            if self.ocr_diagnostics:
                lines.append(("", "\n"))
                separator_width = min(40, self.pane_width - 2)
                lines.append(("class:label", "─" * separator_width + "\n"))
                lines.append(("class:label", "Activity Log:\n"))
                # Get most recent messages and reverse them (newest first)
                all_messages = list(self.ocr_diagnostics)
                messages_to_show = all_messages[-self.activity_log_max_lines:] if len(all_messages) > self.activity_log_max_lines else all_messages
                # Reverse so newest is at top
                # Messages will wrap automatically due to wrap_lines=True
                for msg in reversed(messages_to_show):
                    lines.append(("class:log", f"  {msg}\n"))
            
            return lines
    
    def _get_ffmpeg_pane_text(self):
        """Get text for FFMPEG pane."""
        with self.lock:
            lines = []
            lines.append(("class:header", " FFMPEG Extraction "))
            lines.append(("class:log", " (Press 'e' for early exit, 'q' to quit, Ctrl+C to cancel)\n"))
            lines.append(("", "\n"))
            
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
                lines.append(("class:label", f"Processing File "))
                lines.append(("class:value", f"{self.total_files}/{self.total_files}: All files\n"))
            elif currently_extracting:
                file_index = current_index + 1 if current_index >= 0 else 0
                lines.append(("class:label", f"Processing File "))
                lines.append(("class:value", f"{file_index}/{self.total_files}: {Path(currently_extracting).name}\n"))
            elif self.current_ffmpeg_file and self.current_ffmpeg_file_index >= 0:
                # Show last processed file if we have an index
                file_index = self.current_ffmpeg_file_index + 1
                filename = self.current_ffmpeg_file
                lines.append(("class:label", f"Processing File "))
                lines.append(("class:value", f"{file_index}/{self.total_files}: {Path(filename).name}\n"))
            else:
                lines.append(("class:label", f"Processing File "))
                lines.append(("class:value", f"-/{self.total_files}: -\n"))
            
            lines.append(("", "\n"))
            
            # Show status - use clear status values: decoding, standby, finished
            currently_extracting = None
            for filename, info in self.ffmpeg_extractions.items():
                status = info.get('status', '').lower()
                if status in ['extracting', 'decoding']:
                    currently_extracting = filename
                    break
            
            if all_completed:
                lines.append(("class:label", "Status: "))
                lines.append(("class:success", "finished\n"))
            elif currently_extracting:
                lines.append(("class:label", "Status: "))
                lines.append(("class:value", "decoding\n"))
            else:
                lines.append(("class:label", "Status: "))
                lines.append(("class:value", "standby\n"))
            
            # Show activity log - newest at top (bottom-up scrolling)
            if self.ffmpeg_log:
                lines.append(("", "\n"))
                separator_width = min(40, self.pane_width - 2)
                lines.append(("class:label", "─" * separator_width + "\n"))
                lines.append(("class:label", "Activity Log:\n"))
                # Get most recent messages and reverse them (newest first)
                all_messages = list(self.ffmpeg_log)
                messages_to_show = all_messages[-self.activity_log_max_lines:] if len(all_messages) > self.activity_log_max_lines else all_messages
                # Reverse so newest is at top
                # Messages will wrap automatically due to wrap_lines=True
                for msg in reversed(messages_to_show):
                    lines.append(("class:log", f"  {msg}\n"))
            
            return lines
    
    def _get_summary_pane_text(self):
        """Get text for summary pane showing all matches."""
        with self.lock:
            lines = []
            lines.append(("class:header", " Matches "))
            lines.append(("", "\n"))
            
            completed = len(self.completed_files)
            matched = sum(1 for f in self.completed_files if f.get('matched'))
            
            lines.append(("class:label", f"Completed: "))
            lines.append(("class:value", f"{completed}/{self.total_files}"))
            lines.append(("", "  "))
            lines.append(("class:label", f"Matched: "))
            lines.append(("class:success", f"{matched}\n"))
            
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
                
                lines.append(("", "  "))
                lines.append(("class:label", f"Elapsed: "))
                lines.append(("class:value", f"{time_str}\n"))
            else:
                lines.append(("", "\n"))
            
            lines.append(("", "\n"))
            
            if self.completed_files:
                lines.append(("class:label", "─" * 40 + "\n"))
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
                        lines.append(("class:success", f"✓ "))
                        lines.append(("class:value", f"{Path(file).name}"))
                        if elapsed_str:
                            lines.append(("class:log", elapsed_str))
                        if timestamp_str:
                            lines.append(("class:log", timestamp_str))
                        lines.append(("", "\n"))
                        # Show new filename (fully qualified rename)
                        if new_filename:
                            new_name_display = new_filename[:80] + "..." if len(new_filename) > 80 else new_filename
                            lines.append(("class:log", f"  → {new_name_display}\n"))
                        else:
                            # Fallback: show episode name if new_filename not available
                            ep_name = matched_ep[:70] + "..." if len(matched_ep) > 70 else matched_ep
                            lines.append(("class:log", f"  → {ep_name}\n"))
                        lines.append(("", "\n"))
                    else:
                        # Show filename with X for no match
                        lines.append(("class:error", f"✗ "))
                        lines.append(("class:value", f"{Path(file).name}"))
                        if elapsed_str:
                            lines.append(("class:log", elapsed_str))
                        lines.append(("", "\n"))
                        lines.append(("", "\n"))
            else:
                lines.append(("class:log", "No matches yet...\n"))
            
            return lines
    
    def create_application(self) -> Application:
        """Create the prompt_toolkit application."""
        ocr_control = FormattedTextControl(self._get_ocr_pane_text)
        ocr_window = Window(
            content=ocr_control,
            style='class:ocr-pane',
            wrap_lines=True,  # Enable wrapping so messages wrap to new lines instead of expanding pane
            width=D.exact(self.pane_width)  # Fixed width to prevent pane resizing
        )
        
        ffmpeg_control = FormattedTextControl(self._get_ffmpeg_pane_text)
        ffmpeg_window = Window(
            content=ffmpeg_control,
            style='class:ffmpeg-pane',
            wrap_lines=True,  # Enable wrapping so messages wrap to new lines instead of expanding pane
            width=D.exact(self.pane_width)  # Fixed width to prevent pane resizing
        )
        
        summary_control = FormattedTextControl(self._get_summary_pane_text)
        summary_window = Window(
            content=summary_control,
            style='class:summary-pane',
            wrap_lines=False,
            height=D.exact(self.bottom_section_height)  # Fixed height for bottom section
        )
        
        # Layout: Top half = OCR | FFMPEG (side by side), Bottom half = Summary
        # Use exact heights based on terminal size to ensure 50/50 split
        # Create top section with exact height (OCR and FFMPEG side by side)
        # VSplit divides width equally (50/50) for OCR and FFMPEG
        top_section = VSplit([
            ocr_window,
            ffmpeg_window
        ], height=D.exact(self.top_section_height))  # Exact height for top section
        
        # Bottom section
        bottom_section = summary_window
        
        # HSplit with exact heights ensures 50/50 split
        self.layout = Layout(
            HSplit([
                top_section,
                bottom_section
            ])
        )
        
        kb = KeyBindings()
        
        @kb.add('e')
        @kb.add('E')
        def early_exit(event):
            """Request early exit to preview window."""
            self.early_exit_requested = True
            self.running = False
            event.app.exit()
        
        @kb.add('q')
        @kb.add('Q')
        @kb.add('escape')
        def quit_app(event):
            """Quit the application."""
            self.running = False
            event.app.exit()
        
        @kb.add('c-c')
        def handle_ctrl_c(event):
            """Handle Ctrl-C aggressively - kill everything immediately."""
            # Set force exit flag
            FORCE_EXIT.set()
            self.running = False
            self.early_exit_requested = False  # Ctrl-C means quit, not preview
            
            # Restore terminal before exiting
            try:
                import sys
                import subprocess
                # Try stty sane first (most reliable)
                subprocess.run(['stty', 'sane'], timeout=0.1, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            except Exception:
                # If stty fails, try termios
                try:
                    import termios
                    if sys.stdin.isatty():
                        attrs = termios.tcgetattr(sys.stdin.fileno())
                        attrs[3] = attrs[3] & ~termios.ECHO & ~termios.ICANON
                        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, attrs)
                except Exception:
                    # Last resort: ANSI escape codes
                    try:
                        sys.stdout.write('\x1b[?25h')  # Show cursor
                        sys.stdout.write('\x1b[0m')     # Reset colors
                        sys.stdout.write('\r\n')        # New line
                        sys.stdout.flush()
                    except Exception:
                        pass
            
            # Force immediate exit - bypasses app.exit() which might hang
            os._exit(130)
        
        style = Style.from_dict({
            'header': 'bg:#1e1e2e #a6e3a1 bold',
            'label': '#6c7086',
            'value': '#bac2de',
            'success': '#a6e3a1',
            'error': '#f38ba8',
            'log': '#6c7086',
            'ocr-pane': 'bg:#11111b',
            'ffmpeg-pane': 'bg:#11111b',
            'summary-pane': 'bg:#1e1e2e',
        })
        
        app = Application(
            layout=self.layout,
            key_bindings=kb,
            style=style,
            full_screen=True,
            mouse_support=True,  # Enable mouse support for scrolling
            refresh_interval=0.1,  # Refresh 10 times per second
            erase_when_done=False  # Don't clear screen when done
        )
        
        self.app = app
        return app
    
    def start(self):
        """Start the display in a separate thread."""
        if self.app is None:
            self.create_application()
        
        def run_app():
            self.running = True
            try:
                # Run the app (this blocks until exit is called)
                self.app.run()
            except KeyboardInterrupt:
                self.running = False
                if self.app:
                    try:
                        self.app.exit()
                    except Exception:
                        pass
            except Exception as e:
                import sys
                print(f"Display error: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                self.running = False
            finally:
                self.running = False
        
        self.display_thread = threading.Thread(target=run_app, daemon=True)  # Daemon thread
        self.display_thread.start()
        
        # Give it a moment to start and render
        time.sleep(0.3)
    
    def stop(self):
        """Stop the display."""
        # Check if force exit is requested - if so, exit immediately
        if FORCE_EXIT.is_set():
            # Restore terminal before exiting
            try:
                import sys
                import subprocess
                # Try stty sane first (most reliable)
                subprocess.run(['stty', 'sane'], timeout=0.1, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            except Exception:
                # If stty fails, try termios
                try:
                    import termios
                    if sys.stdin.isatty():
                        attrs = termios.tcgetattr(sys.stdin.fileno())
                        attrs[3] = attrs[3] & ~termios.ECHO & ~termios.ICANON
                        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, attrs)
                except Exception:
                    # Last resort: ANSI escape codes
                    try:
                        sys.stdout.write('\x1b[?25h')  # Show cursor
                        sys.stdout.write('\x1b[0m')     # Reset colors
                        sys.stdout.write('\r\n')        # New line
                        sys.stdout.flush()
                    except Exception:
                        pass
            os._exit(130)
        
        self.running = False
        if self.app:
            try:
                # Force exit the application
                self.app.exit()
            except Exception:
                pass
            # Don't wait for thread - it's a daemon thread and will exit when main thread exits
            # Waiting can cause hangs if the app.run() is blocked
    
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
        
        if self.app:
            self.app.invalidate()
    
    def add_ocr_completed(self, file: str, matched: Optional[str], status: str = ""):
        """Add a completed OCR file to the OCR pane."""
        with self.lock:
            self.ocr_completed.append({
                'file': file,
                'matched': matched,
                'status': status
            })
        
        if self.app:
            self.app.invalidate()
    
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
            
            # Don't add message to log here - it's handled by add_ffmpeg_diagnostic
        
        if self.app:
            self.app.invalidate()
    
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
        
        if self.app:
            self.app.invalidate()
    
    def add_ocr_diagnostic(self, message: str):
        """Add a diagnostic message to OCR pane."""
        with self.lock:
            self.ocr_diagnostics.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        
        if self.app:
            self.app.invalidate()
    
    def add_ffmpeg_diagnostic(self, message: str):
        """Add a diagnostic message to FFMPEG pane."""
        with self.lock:
            self.ffmpeg_log.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        
        if self.app:
            self.app.invalidate()
    
    def set_config_info(self, config_text: str):
        """Set configuration information to display."""
        with self.lock:
            self.config_info = config_text
        
        if self.app:
            self.app.invalidate()
    


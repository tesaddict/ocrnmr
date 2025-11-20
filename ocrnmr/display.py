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


class OCRProgressDisplay:
    """Display progress for OCR processing."""
    
    def __init__(self):
        """Initialize the display."""
        self.lock = threading.Lock()
        
        # State
        self.current_file: Optional[str] = None
        self.current_file_index = 0
        self.total_files = 0
        self.status = "Waiting..."
        self.message = ""
        self.log: deque = deque(maxlen=100)
        
        # Results
        self.completed_files: List[Dict] = []
        
        # Timing
        self.ocr_start_time: Optional[float] = None
        self.ocr_end_time: Optional[float] = None
        
        # UI
        self.console = Console()
        self.live: Optional[Live] = None
        self.running = False
        self.early_exit_requested = False
        self.activity_log_max_lines = 10
        
    def update_status(self, file: str, index: int, total: int, status: str, message: str = ""):
        """Update processing status."""
        with self.lock:
            self.current_file = file
            self.current_file_index = index
            self.total_files = total
            self.status = status
            if message:
                self.message = message
                
    def add_log(self, message: str):
        """Add a message to the activity log."""
        with self.lock:
            self.log.append(f"[{time.strftime('%H:%M:%S')}] {message}")
            
    def add_completed_file(self, file: str, matched: Optional[str], success: bool, new_filename: Optional[str] = None, match_timestamp: Optional[float] = None):
        """Add a completed file to the summary."""
        with self.lock:
            self.completed_files.append({
                'file': file,
                'matched': matched,
                'success': success,
                'new_filename': new_filename,
                'match_timestamp': match_timestamp
            })

    def show_error(self, error_message: str):
        """Display an error message."""
        self.add_log(f"ERROR: {error_message}")
        
    def _get_status_pane_content(self) -> Panel:
        """Get content for status pane."""
        with self.lock:
            content_lines = []
            content_lines.append(Text(" Processing Status ", style="bold green on dark_blue"))
            content_lines.append(Text(" (Press Ctrl+C to cancel)\n", style="dim"))
            content_lines.append(Text("\n"))
            
            # Current file
            if self.current_file:
                content_lines.append(Text("File: ", style="dim"))
                content_lines.append(Text(f"{self.current_file_index + 1}/{self.total_files}: {Path(self.current_file).name}\n", style="bright_white"))
            else:
                content_lines.append(Text("File: ", style="dim"))
                content_lines.append(Text(f"-/{self.total_files}: -\n", style="bright_white"))
                
            content_lines.append(Text("\n"))
            
            # Status
            content_lines.append(Text("Status: ", style="dim"))
            content_lines.append(Text(f"{self.status}\n", style="bright_white"))
            
            if self.message:
                content_lines.append(Text("\n"))
                content_lines.append(Text(f"{self.message}\n", style="dim"))
                
            combined_text = Text()
            for line in content_lines:
                combined_text.append(line)
                
            return Panel(
                combined_text,
                title="Status",
                border_style="green"
            )

    def _get_log_pane_content(self) -> Panel:
        """Get content for log pane."""
        with self.lock:
            content_lines = []
            
            # Show activity log - newest at top
            if self.log:
                all_messages = list(self.log)
                messages_to_show = all_messages[-self.activity_log_max_lines:] if len(all_messages) > self.activity_log_max_lines else all_messages
                for msg in reversed(messages_to_show):
                    content_lines.append(Text(f"{msg}\n", style="dim"))
            else:
                content_lines.append(Text("No activity yet...\n", style="dim"))
                
            combined_text = Text()
            for line in content_lines:
                combined_text.append(line)
                
            return Panel(
                combined_text,
                title="Activity Log",
                border_style="blue"
            )

    def _get_summary_pane_content(self) -> Panel:
        """Get content for summary pane."""
        with self.lock:
            content_lines = []
            
            completed = len(self.completed_files)
            matched = sum(1 for f in self.completed_files if f.get('matched'))
            
            content_lines.append(Text("Completed: ", style="dim"))
            content_lines.append(Text(f"{completed}/{self.total_files}", style="bright_white"))
            content_lines.append(Text("  ", style="dim"))
            content_lines.append(Text("Matched: ", style="dim"))
            content_lines.append(Text(f"{matched}\n", style="green"))
            
            # Elapsed time
            if self.ocr_start_time is not None:
                end_time = self.ocr_end_time if self.ocr_end_time else time.time()
                elapsed = end_time - self.ocr_start_time
                
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
            
            content_lines.append(Text("\n"))
            
            if self.completed_files:
                content_lines.append(Text("─" * 40 + "\n", style="dim"))
                for result in reversed(self.completed_files):
                    file = result.get('file', 'unknown')
                    matched_ep = result.get('matched')
                    new_filename = result.get('new_filename')
                    match_timestamp = result.get('match_timestamp')
                    
                    timestamp_str = ""
                    if match_timestamp is not None:
                        minutes = int(match_timestamp // 60)
                        seconds = int(match_timestamp % 60)
                        timestamp_str = f" ({minutes}:{seconds:02d})"
                    
                    if matched_ep:
                        content_lines.append(Text("✓ ", style="green"))
                        content_lines.append(Text(f"{Path(file).name}", style="bright_white"))
                        if timestamp_str:
                            content_lines.append(Text(timestamp_str, style="dim"))
                        content_lines.append(Text("\n"))
                        if new_filename:
                            new_name_display = new_filename[:80] + "..." if len(new_filename) > 80 else new_filename
                            content_lines.append(Text(f"  → {new_name_display}\n", style="dim"))
                        else:
                            ep_name = matched_ep[:70] + "..." if len(matched_ep) > 70 else matched_ep
                            content_lines.append(Text(f"  → {ep_name}\n", style="dim"))
                    else:
                        content_lines.append(Text("✗ ", style="red"))
                        content_lines.append(Text(f"{Path(file).name}\n", style="bright_white"))
                    content_lines.append(Text("\n"))
            else:
                content_lines.append(Text("No matches yet...\n", style="dim"))
                
            combined_text = Text()
            for line in content_lines:
                combined_text.append(line)
                
            return Panel(
                combined_text,
                title="Matches",
                border_style="cyan"
            )

    def _create_layout(self) -> Layout:
        """Create the layout."""
        layout = Layout()
        layout.split_column(
            Layout(self._get_status_pane_content(), size=10),
            Layout(self._get_log_pane_content(), size=12),
            Layout(self._get_summary_pane_content())
        )
        return layout

    def start(self):
        """Start the display."""
        def run_display():
            self.running = True
            try:
                self.console.clear()
                with Live(self._create_layout(), refresh_per_second=4, screen=False, console=self.console) as live:
                    self.live = live
                    while self.running:
                        live.update(self._create_layout())
                        time.sleep(0.25)
            except KeyboardInterrupt:
                self.running = False
            except Exception:
                self.running = False
            finally:
                self.running = False
        
        self.display_thread = threading.Thread(target=run_display, daemon=True)
        self.display_thread.start()
        time.sleep(0.5)

    def stop(self):
        """Stop the display."""
        self.running = False

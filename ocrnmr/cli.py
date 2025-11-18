#!/usr/bin/env python3
"""Command-line interface for OCR NMR application."""

import sys
from pathlib import Path

# Add parent directory to path for direct script execution
# This allows the script to be run as: python3 ocrnmr/cli.py
script_dir = Path(__file__).parent.parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import display module
sys.path.insert(0, str(Path(__file__).parent.parent))
from ocrnmr_display import OCRProgressDisplay

from ocrnmr.ocr import PipelinedOCRProcessor
from ocrnmr.tmdb_client import TMDBClient
from ocrnmr.filename import sanitize_filename
from ocrnmr.exit_flag import FORCE_EXIT

logger = logging.getLogger(__name__)

# Suppress logging output when using prompt_toolkit display
# Logging messages interfere with the display, so we suppress them
# and route important messages through the display system instead
# Suppress specific loggers that might output during processing
def _suppress_logging():
    """Suppress logging output that interferes with prompt_toolkit."""
    null_handler = logging.NullHandler()
    for logger_name in ['ocrnmr.ocr.processor', 'ocrnmr.ocr', 'ocrnmr']:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)
        logger.handlers = [null_handler]  # Replace all handlers
        logger.propagate = False  # Prevent propagation to root logger

# Call immediately to suppress logging before any modules are imported
_suppress_logging()


def load_config(config_path: Path) -> Dict:
    """Load configuration from JSON file."""
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)


def get_episode_titles(config: Dict, display: Optional[OCRProgressDisplay] = None) -> List[str]:
    """Get episode titles from TMDB and/or config."""
    show_config = config.get("show", {})
    episode_titles = []
    
    show_name = show_config.get("show_name")
    season = show_config.get("season")
    tmdb_api_key = show_config.get("tmdb_api_key")
    manual_titles = show_config.get("episode_titles", [])
    
    # Try to fetch from TMDB if show_name and season are provided
    if show_name and season:
        try:
            # Get API key from config or environment
            api_key = tmdb_api_key
            if not api_key:
                import os
                api_key = os.getenv('TMDB_API_KEY')
            
            if api_key:
                if display:
                    display.add_ocr_diagnostic(f"Fetching episode titles from TMDB for {show_name} Season {season}...")
                client = TMDBClient(api_key=api_key)
                tmdb_titles = client.get_episode_titles(show_name, season)
                if tmdb_titles:
                    episode_titles.extend(tmdb_titles)
                    if display:
                        display.add_ocr_diagnostic(f"Found {len(tmdb_titles)} episodes from TMDB")
                else:
                    if display:
                        display.add_ocr_diagnostic("No episodes found in TMDB")
            else:
                if display:
                    display.add_ocr_diagnostic("TMDB API key not provided, skipping TMDB lookup")
        except Exception as e:
            if display:
                display.add_ocr_diagnostic(f"Error fetching from TMDB: {e}")
                display.add_ocr_diagnostic("Continuing with manual episode titles only...")
    
    # Append manual titles
    if manual_titles:
        episode_titles.extend(manual_titles)
        if display:
            display.add_ocr_diagnostic(f"Added {len(manual_titles)} manual episode titles")
    
    if not episode_titles:
        if display:
            display.add_ocr_diagnostic("Error: No episode titles found")
        print("Error: No episode titles found. Please provide show_name/season for TMDB or manual episode_titles in config.")
        sys.exit(1)
    
    return episode_titles


def get_episode_info(config: Dict, display: Optional[OCRProgressDisplay] = None) -> Dict[str, Tuple[int, int]]:
    """
    Get episode information (season, episode number) from TMDB.
    Returns a dict mapping episode_title -> (season, episode_number).
    """
    show_config = config.get("show", {})
    episode_info = {}  # episode_title -> (season, episode_number)
    
    show_name = show_config.get("show_name")
    season = show_config.get("season")
    tmdb_api_key = show_config.get("tmdb_api_key")
    manual_titles = show_config.get("episode_titles", [])
    
    # Try to fetch from TMDB if show_name and season are provided
    if show_name and season:
        try:
            # Get API key from config or environment
            api_key = tmdb_api_key
            if not api_key:
                import os
                api_key = os.getenv('TMDB_API_KEY')
            
            if api_key:
                if display:
                    display.add_ocr_diagnostic(f"Fetching episode info from TMDB for {show_name} Season {season}...")
                client = TMDBClient(api_key=api_key)
                tmdb_episodes = client.get_episode_info(show_name, season)
                if tmdb_episodes:
                    # Map episode titles to (season, episode_number)
                    for ep_num, ep_title in tmdb_episodes:
                        episode_info[ep_title] = (season, ep_num)
                    if display:
                        display.add_ocr_diagnostic(f"Found {len(tmdb_episodes)} episodes from TMDB")
        except Exception as e:
            if display:
                display.add_ocr_diagnostic(f"Error fetching episode info from TMDB: {e}")
    
    # For manual titles, assign sequential episode numbers starting from 1
    # Only assign if not already in episode_info (TMDB takes precedence)
    for idx, title in enumerate(manual_titles, start=1):
        if title not in episode_info:
            episode_info[title] = (season, idx)
    
    return episode_info


def process_video_files(
    input_directory: Path,
    episode_titles: List[str],
    ocr_config: Dict,
    display: OCRProgressDisplay,
    dry_run: bool = False,
    processor_ref: Optional[Dict] = None,
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
    
    # Create status callback function to update display
    def status_callback(event_type: str, file_index: Optional[int], file_path: Optional[str], 
                       status: str, message: str, source: str = 'ocr', match_timestamp: Optional[float] = None) -> None:
        """Callback function to update display based on processor events.
        
        Args:
            event_type: Type of event ('ffmpeg_start', 'ffmpeg_complete', 'ocr_start', etc.)
            file_index: Index of the file (0-based)
            file_path: Path to the file
            status: Status string ('decoding', 'processing', 'finished', etc.)
            message: Message to display
            source: Source of the message ('ffmpeg' or 'ocr') - determines which pane gets the diagnostic
        """
        if file_path is None:
            return
        
        file_path_str = file_path
        idx = file_index if file_index is not None else 0
        
        if event_type == 'ffmpeg_start':
            # Use provided file_index, or extract from path if not provided
            if file_index is None:
                try:
                    idx = video_files.index(Path(file_path))
                except (ValueError, IndexError):
                    idx = 0
            else:
                idx = file_index
            display.update_ffmpeg_status(
                filename=file_path_str,
                status=status,
                file_index=idx,
                message=message
            )
            # Route to FFMPEG diagnostics
            if source == 'ffmpeg':
                display.add_ffmpeg_diagnostic(message)
            else:
                display.add_ocr_diagnostic(message)
        
        elif event_type == 'ffmpeg_complete':
            display.update_ffmpeg_status(
                filename=file_path_str,
                status=status,
                file_index=idx,
                message=message
            )
            # Route to FFMPEG diagnostics
            if source == 'ffmpeg':
                display.add_ffmpeg_diagnostic(message)
            else:
                display.add_ocr_diagnostic(message)
        
        elif event_type == 'ffmpeg_error':
            display.update_ffmpeg_status(
                filename=file_path_str,
                status=status,
                file_index=idx,
                message=message
            )
            # Route to FFMPEG diagnostics
            if source == 'ffmpeg':
                display.add_ffmpeg_diagnostic(message)
            else:
                display.add_ocr_diagnostic(message)
        
        elif event_type == 'ocr_start':
            # Track start time for this file when OCR processing begins
            with display.lock:
                if file_path_str and file_path_str not in display.file_start_times:
                    display.file_start_times[file_path_str] = time.time()
            
            display.update_ocr_status(
                file=file_path_str,
                index=idx,
                total=total_files,
                status=status
            )
            # Route to OCR diagnostics
            if source == 'ocr':
                display.add_ocr_diagnostic(message)
            else:
                display.add_ffmpeg_diagnostic(message)
        
        elif event_type == 'ocr_complete':
            display.update_ocr_status(
                file=file_path_str,
                index=idx,
                total=total_files,
                status=status
            )
            # Route to OCR diagnostics
            if source == 'ocr':
                display.add_ocr_diagnostic(message)
            else:
                display.add_ffmpeg_diagnostic(message)
        
        elif event_type == 'match_found':
            # Extract matched episode from message
            if message.startswith("Match found: "):
                matched_episode = message.replace("Match found: ", "")
                # Generate new filename using episode_info
                new_filename = None
                if matched_episode and episode_info and matched_episode in episode_info:
                    ep_season, ep_num = episode_info[matched_episode]
                    file_path = Path(file_path_str)
                    ext = file_path.suffix
                    new_filename = f"{show_name} - S{ep_season:02d}E{ep_num:02d} - {matched_episode}{ext}"
                    new_filename = sanitize_filename(new_filename)
                # Route to OCR diagnostics
                if source == 'ocr':
                    display.add_ocr_diagnostic(message)
                else:
                    display.add_ffmpeg_diagnostic(message)
                display.add_completed_file(file_path_str, matched_episode, True, new_filename=new_filename, match_timestamp=match_timestamp)
            elif message.startswith("No match found"):
                # Route to OCR diagnostics
                if source == 'ocr':
                    display.add_ocr_diagnostic(message)
                else:
                    display.add_ffmpeg_diagnostic(message)
                display.add_completed_file(file_path_str, None, False, new_filename=None, match_timestamp=None)
            else:
                # Route based on source
                if source == 'ocr':
                    display.add_ocr_diagnostic(message)
                else:
                    display.add_ffmpeg_diagnostic(message)
    
    # Ensure logging is suppressed (in case processor module was imported after initial suppression)
    _suppress_logging()
    
    # Initialize processor with memory limit from config (default 1GB)
    memory_limit = ocr_config.get("memory_limit_bytes", 1073741824)
    processor = PipelinedOCRProcessor(memory_limit_bytes=memory_limit)
    
    # Store processor reference if provided (for cleanup on interrupt)
    if processor_ref is not None and isinstance(processor_ref, dict):
        processor_ref['processor'] = processor
    
    # Check for early exit function
    def should_continue() -> bool:
        """Check if processing should continue."""
        return not display.early_exit_requested
    
    # Mark OCR start time
    import time
    display.ocr_start_time = time.time()
    
    # Process files using pipelined processor
    try:
        matches = processor.process_files(
            video_files=video_files,
            episode_titles=episode_titles,
            config=ocr_config,
            status_callback=status_callback,
            should_continue=should_continue
        )
    except KeyboardInterrupt:
        # Force exit immediately - signal handler should have already killed everything
        FORCE_EXIT.set()
        os._exit(130)
    finally:
        # Mark OCR end time
        display.ocr_end_time = time.time()
    
    # All processing complete
    display.update_ocr_status(
        file="",
        index=total_files,
        total=total_files,
        status="finished"
    )
    
    return matches


def generate_rename_preview(
    matches: List[Tuple[Path, Optional[str]]],
    show_name: str,
    season: int,
    episode_info: Dict[str, Tuple[int, int]]
) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """
    Generate rename preview data.
    
    Uses episode numbers from TMDB (via episode_info dict) when available.
    Falls back to sequential numbering if episode info is not available.
    
    Args:
        matches: List of (file_path, matched_episode_title) tuples
        show_name: Name of the show
        season: Season number from config
        episode_info: Dict mapping episode_title -> (season, episode_number)
    
    Returns:
        List of tuples (original_name, new_name, episode_name)
    """
    rename_preview = []
    
    for file_path, matched_episode in matches:
        original_name = file_path.name
        ext = file_path.suffix
        
        if matched_episode:
            # Look up episode info from TMDB
            if matched_episode in episode_info:
                ep_season, ep_num = episode_info[matched_episode]
                new_name = f"{show_name} - S{ep_season:02d}E{ep_num:02d} - {matched_episode}{ext}"
            else:
                # Fall back to config season and sequential episode numbering
                # Count unique episodes seen so far
                unique_episodes = set(ep for _, ep, _ in rename_preview if ep)
                ep_num = len(unique_episodes) + 1
                new_name = f"{show_name} - S{season:02d}E{ep_num:02d} - {matched_episode}{ext}"
            
            new_name = sanitize_filename(new_name)
            rename_preview.append((original_name, new_name, matched_episode))
        else:
            rename_preview.append((original_name, None, None))
    
    return rename_preview


def display_rename_preview(rename_preview: List[Tuple[str, Optional[str], Optional[str]]], 
                          display: OCRProgressDisplay, show_name: str, season: int):
    """Display rename preview using rich console formatting."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    
    console = Console()
    matched_count = sum(1 for _, new_name, _ in rename_preview if new_name)
    total_count = len(rename_preview)
    
    # Stop the main display cleanly (this will exit the main app)
    display.stop()
    
    # Wait a moment for the display thread to fully stop
    import time
    time.sleep(0.2)
    
    # Clear screen and print preview
    console.print()  # Add a blank line
    console.print(Panel(
        f"[bold cyan]{show_name}[/bold cyan] [dim]Season {season:02d}[/dim]\n"
        f"[green]{matched_count}[/green] of [yellow]{total_count}[/yellow] files matched",
        title="[bold]Rename Preview[/bold]",
        border_style="cyan"
    ))
    console.print()
    
    # Create a table for the preview
    table = Table(show_header=True, header_style="bold cyan", border_style="dim")
    table.add_column("Original Filename", style="yellow", overflow="fold")
    table.add_column("New Filename", style="green", overflow="fold")
    table.add_column("Episode", style="dim", overflow="fold")
    
    for original, new_name, episode in rename_preview:
        if new_name:
            ep_display = episode[:60] + "..." if episode and len(episode) > 60 else (episode or "")
            table.add_row(original, new_name, ep_display)
        else:
            table.add_row(original, "[red][No match found][/red]", "")
    
    console.print(table)
    console.print()


def execute_renames(rename_preview: List[Tuple[str, Optional[str], Optional[str]]], input_directory: Path):
    """Execute file renames."""
    renamed_count = 0
    for original, new_name, _ in rename_preview:
        if new_name:
            original_path = input_directory / original
            new_path = input_directory / new_name
            
            # Skip if already renamed
            if original_path.name == new_name:
                continue
            
            # Check if target already exists
            if new_path.exists():
                print(f"⚠ Skipping {original}: target already exists")
                continue
            
            try:
                original_path.rename(new_path)
                renamed_count += 1
            except Exception as e:
                print(f"Error renaming {original}: {e}")
    
    print(f"\n✓ Renamed {renamed_count} files")


def _restore_terminal():
    """Restore terminal to normal mode before exiting."""
    try:
        import sys
        # Try to reset terminal using stty command (most reliable)
        import subprocess
        subprocess.run(['stty', 'sane'], timeout=0.1, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    except Exception:
        # If stty fails, try termios
        try:
            import termios
            if sys.stdin.isatty():
                # Get current attributes and restore them (this should reset to sane defaults)
                attrs = termios.tcgetattr(sys.stdin.fileno())
                # Reset to sane defaults
                attrs[3] = attrs[3] & ~termios.ECHO & ~termios.ICANON  # Clear echo and canonical mode flags
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


def _aggressive_sigint_handler(signum, frame):
    """Aggressive SIGINT handler that kills everything and exits immediately."""
    # Set force exit flag
    FORCE_EXIT.set()
    
    # Kill all FFMPEG processes immediately
    try:
        subprocess.run(['pkill', '-9', 'ffmpeg'], timeout=0.5, capture_output=True)
    except Exception:
        pass
    
    # Restore terminal before exiting
    _restore_terminal()
    
    # Force immediate exit - bypasses all cleanup
    os._exit(130)  # Standard exit code for Ctrl-C


def main():
    """Main entry point."""
    # Register aggressive SIGINT handler for immediate exit
    signal.signal(signal.SIGINT, _aggressive_sigint_handler)
    
    parser = argparse.ArgumentParser(
        description="OCR-based episode renaming tool for TV shows"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview renames without executing"
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config).expanduser()
    config = load_config(config_path)
    
    # Logging is already suppressed at module level above
    # This ensures no logger output interferes with prompt_toolkit display
    
    # Initialize display
    display = OCRProgressDisplay()
    display.start()
    
    try:
        # Get episode titles
        episode_titles = get_episode_titles(config, display)
        display.add_ocr_diagnostic(f"Total episode titles: {len(episode_titles)}")
        
        # Get input directory
        show_config = config.get("show", {})
        input_directory = Path(show_config.get("input_directory", ".")).expanduser()
        if not input_directory.exists():
            display.add_ocr_diagnostic(f"Error: Input directory not found: {input_directory}")
            print(f"Error: Input directory not found: {input_directory}")
            sys.exit(1)
        
        # Get show name and season for filename generation
        show_name = show_config.get("show_name", "Unknown Show")
        season = show_config.get("season", 1)
        
        # Process video files
        ocr_config = config.get("ocr", {})
        early_exit = False
        processor = None  # Initialize to None so it's in scope for exception handler
        try:
            # Get episode info early so we can use it in status callback
            episode_info = get_episode_info(config, display)
            
            matches = process_video_files(input_directory, episode_titles, ocr_config, display, args.dry_run, processor_ref=None, episode_info=episode_info, show_name=show_name)
            # Check if early exit was requested
            early_exit = display.early_exit_requested
            if early_exit:
                display.add_ocr_diagnostic("Early exit requested - showing preview with current matches")
        except KeyboardInterrupt:
            # Force exit immediately - signal handler should have already killed everything
            FORCE_EXIT.set()
            _restore_terminal()
            os._exit(130)
        
        # Generate rename preview (even if early exit or interrupted)
        # episode_info was already fetched above
        rename_preview = generate_rename_preview(matches, show_name, season, episode_info)
        
        # Display preview window (always show, even on early exit)
        display_rename_preview(rename_preview, display, show_name, season)
        
        # Execute renames unless dry-run
        if not args.dry_run:
            execute_renames(rename_preview, input_directory)
        
    except KeyboardInterrupt:
        # Force exit immediately - signal handler should have already killed everything
        FORCE_EXIT.set()
        _restore_terminal()
        os._exit(130)
    finally:
        # Check force exit before cleanup
        if FORCE_EXIT.is_set():
            _restore_terminal()
            os._exit(130)
        # Clean up display
        try:
            display.stop()
        except Exception:
            pass
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


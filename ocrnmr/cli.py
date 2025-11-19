#!/usr/bin/env python3
"""Command-line interface for OCR NMR application."""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

# Add parent directory to path for direct script execution
# This allows the script to be run as: python3 ocrnmr/cli.py
script_dir = Path(__file__).parent.parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# All imports at the top - simpler and more reliable
from ocrnmr.display import OCRProgressDisplay
from ocrnmr.episode_fetcher import fetch_episodes
from ocrnmr.video_processor import process_video_files
from ocrnmr.rename_executor import generate_rename_preview, display_rename_preview, execute_renames
from ocrnmr.exit_flag import FORCE_EXIT

# Rich imports for console output
from rich.console import Console

logger = logging.getLogger(__name__)

# Suppress logging output when using rich display
# Logging messages interfere with the display, so we suppress them
# and route important messages through the display system instead
# Suppress specific loggers that might output during processing
def _suppress_logging():
    """Suppress logging output that interferes with rich display."""
    null_handler = logging.NullHandler()
    for logger_name in ['ocrnmr.ocr.processor', 'ocrnmr.ocr', 'ocrnmr']:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)
        logger.handlers = [null_handler]  # Replace all handlers
        logger.propagate = False  # Prevent propagation to root logger






def _restore_terminal():
    """Restore terminal to normal mode before exiting."""
    try:
        # Try to reset terminal using stty command (most reliable)
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
        description="OCR-based episode renaming tool for TV shows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with TMDB (no config file needed):
  ocrnmr --show "Star Trek: The Next Generation" --season 1 --input ~/videos/season1/

  # With custom episodes (episodes file required):
  ocrnmr --show "Dragon Ball Z Kai" --season 5 --input ~/videos/ --episodes-file episodes.json

  # Override OCR settings:
  ocrnmr --show "Stargate SG-1" --season 2 --input ~/videos/ --max-dimension 1200 --duration 900

  # Dry run to preview changes:
  ocrnmr --show "The Office" --season 1 --input ~/videos/ --dry-run

Episodes File Format:
  The episodes file (--episodes-file) is optional and only needed for custom episodes when TMDB
  doesn't have complete information. It should contain:
  
  {
    "episodes": [
      {"episode": 1, "title": "Episode Title 1"},
      {"episode": 1, "title": "Episode Title Variant"},
      {"episode": 2, "title": "Episode Title 2"}
    ]
  }
  
  All other settings (OCR parameters, show info) come from CLI arguments.
        """
    )
    
    
    # Required arguments
    parser.add_argument(
        "--show",
        type=str,
        required=True,
        help="TV show name (required)"
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="Season number (required)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory path containing video files (required)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--episodes-file",
        type=str,
        default=None,
        help="Path to JSON file containing custom episodes array (optional, only needed when TMDB doesn't have complete info)"
    )
    parser.add_argument(
        "--tmdb-key",
        type=str,
        default=None,
        help="TMDB API key (optional, can also use TMDB_API_KEY environment variable)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview renames without executing"
    )
    
    # OCR settings (all optional with defaults)
    parser.add_argument(
        "--duration",
        type=int,
        default=600,
        help="Maximum duration to process in seconds (default: 600 = 10 minutes, use 0 for full video)"
    )
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=2.0,
        help="Interval between frames in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.65,
        help="Episode matching threshold 0.0-1.0 (default: 0.65, lower = more lenient)"
    )
    parser.add_argument(
        "--hwaccel",
        type=str,
        default=None,
        choices=['videotoolbox', 'vaapi', 'd3d11va', 'dxva2'],
        help="Hardware acceleration: videotoolbox (macOS), vaapi (Linux), d3d11va/dxva2 (Windows). Default: None (software)"
    )
    
    args = parser.parse_args()
    
    # Suppress logging (modules already imported at top)
    _suppress_logging()
    
    # Build OCR config from CLI arguments
    ocr_config = {
        "duration": args.duration if args.duration > 0 else None,
        "frame_interval": args.frame_interval,
        "match_threshold": args.match_threshold,
        "hwaccel": args.hwaccel,
    }
    
    # Validate episodes file exists if provided
    if args.episodes_file:
        episodes_path = Path(args.episodes_file).expanduser()
        if not episodes_path.exists():
            console = Console(stderr=True)
            console.print(f"[red]Error:[/red] Episodes file not found: {episodes_path}")
            sys.exit(1)
    
    # Logging is already suppressed at module level above
    # This ensures no logger output interferes with rich display
    
    # Initialize display with error handling
    display = None
    try:
        display = OCRProgressDisplay()
        display.start()
    except Exception as e:
        # Use rich console for error output
        console = Console(stderr=True)
        console.print(f"[red]ERROR:[/red] Failed to start display: {e}")
        # Continue without display if it fails
        console.print("[yellow]WARNING:[/yellow] Continuing without display")
    
    try:
        # Get input directory
        input_directory = Path(args.input).expanduser()
        if not input_directory.exists():
            error_msg = f"Error: Input directory not found: {input_directory}"
            if display:
                display.show_error(error_msg)
                display.add_ocr_diagnostic(error_msg)
                time.sleep(2.0)
            else:
                console = Console(stderr=True)
                console.print(f"[red]{error_msg}[/red]")
            sys.exit(1)
        
        # Get show name and season for filename generation
        show_name = args.show
        season = args.season
        
        # Fetch episodes from TMDB and/or episodes file
        episodes_file_path = Path(args.episodes_file).expanduser() if args.episodes_file else None
        episode_titles, episode_info = fetch_episodes(
            show_name=show_name,
            season=season,
            tmdb_api_key=args.tmdb_key,
            episodes_file=episodes_file_path,
            display=display
        )
        
        if display:
            display.add_ocr_diagnostic(f"Total episode titles: {len(episode_titles)}")
        
        early_exit = False
        processor = None  # Initialize to None so it's in scope for exception handler
        try:
            
            matches = process_video_files(input_directory, episode_titles, ocr_config, display, episode_info=episode_info, show_name=show_name)
            # Check if early exit was requested
            early_exit = display.early_exit_requested if display else False
            if early_exit and display:
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
        if display:
            try:
                display.stop()
            except Exception as e:
                # Silently ignore errors when stopping display
                pass
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


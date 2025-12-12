#!/usr/bin/env python3
"""Command-line interface for OCR NMR application."""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path for direct script execution
script_dir = Path(__file__).parent.parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from ocrnmr.display import OCRProgressDisplay
from ocrnmr.episode_fetcher import fetch_episodes
from ocrnmr.processor import process_video_files
from ocrnmr.rename_executor import generate_rename_preview, display_rename_preview, execute_renames
from ocrnmr.profiler import profiler

# Rich imports for console output
from rich.console import Console

logger = logging.getLogger(__name__)

def _suppress_logging():
    """Suppress logging output that interferes with rich display."""
    null_handler = logging.NullHandler()
    for logger_name in ['ocrnmr.ocr.processor', 'ocrnmr.ocr', 'ocrnmr']:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)
        logger.handlers = [null_handler]
        logger.propagate = False

def main():
    """Main entry point."""
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
        """
    )
    
    # Required arguments
    parser.add_argument("--show", type=str, required=True, help="TV show name (required)")
    parser.add_argument("--season", type=int, required=True, help="Season number (required)")
    parser.add_argument("--input", type=str, required=True, help="Input directory path containing video files (required)")
    
    # Optional arguments
    parser.add_argument("--episodes-file", type=str, default=None, help="Path to JSON file containing custom episodes array")
    parser.add_argument("--tmdb-key", type=str, default=None, help="TMDB API key")
    parser.add_argument("--dry-run", action="store_true", help="Preview renames without executing")
    
    # OCR settings
    parser.add_argument("--duration", type=int, default=600, help="Maximum duration to process in seconds (default: 600)")
    parser.add_argument("--frame-interval", type=float, default=2.0, help="Interval between frames in seconds (default: 2.0)")
    parser.add_argument("--match-threshold", type=float, default=0.65, help="Episode matching threshold 0.0-1.0 (default: 0.65)")
    parser.add_argument("--max-dimension", type=int, default=800, help="Maximum width/height for OCR (default: 800)")
    parser.add_argument("--min-dimension", type=int, default=320, help="Minimum width/height for OCR first pass (default: 320)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for OCR processing (default: 32)")
    parser.add_argument("--ocr-device", type=str, default="auto", choices=["auto", "gpu", "cpu"], help="Force OCR device usage (default: auto)")
    parser.add_argument("--save-match-screenshot", action="store_true", help="Save screenshot of matched frame to matched_screenshots directory")
    parser.add_argument("--profile", type=str, default=None, help="Enable profiling and write to specified JSON file")
    parser.add_argument("--hwaccel", type=str, default=None, choices=['videotoolbox', 'vaapi', 'd3d11va', 'dxva2', 'cuda', 'auto'], help="FFmpeg Hardware acceleration")
    
    args = parser.parse_args()
    
    _suppress_logging()
    
    if args.profile:
        profiler.enable(args.profile)
    
    # Map ocr-device to boolean or None
    ocr_gpu = None
    
    # If hwaccel is cuda and ocr-device is auto, default to gpu
    if args.hwaccel == 'cuda' and args.ocr_device == 'auto':
        ocr_gpu = True
    elif args.ocr_device == "gpu":
        ocr_gpu = True
    elif args.ocr_device == "cpu":
        ocr_gpu = False
    
    ocr_config = {
        "duration": args.duration if args.duration > 0 else None,
        "frame_interval": args.frame_interval,
        "match_threshold": args.match_threshold,
        "max_dimension": args.max_dimension,
        "min_dimension": args.min_dimension,
        "batch_size": args.batch_size,
        "hwaccel": args.hwaccel,
        "gpu": ocr_gpu,
        "save_match_screenshot": args.save_match_screenshot,
    }
    
    if args.episodes_file:
        episodes_path = Path(args.episodes_file).expanduser()
        if not episodes_path.exists():
            console = Console(stderr=True)
            console.print(f"[red]Error:[/red] Episodes file not found: {episodes_path}")
            sys.exit(1)
    
    display = None
    try:
        display = OCRProgressDisplay()
        display.start()
    except Exception as e:
        console = Console(stderr=True)
        console.print(f"[red]ERROR:[/red] Failed to start display: {e}")
        console.print("[yellow]WARNING:[/yellow] Continuing without display")
    
    try:
        input_directory = Path(args.input).expanduser()
        if not input_directory.exists():
            error_msg = f"Error: Input directory not found: {input_directory}"
            if display:
                display.show_error(error_msg)
                time.sleep(2.0)
            else:
                console = Console(stderr=True)
                console.print(f"[red]{error_msg}[/red]")
            sys.exit(1)
        
        show_name = args.show
        season = args.season
        
        episodes_file_path = Path(args.episodes_file).expanduser() if args.episodes_file else None
        episode_titles, episode_info = fetch_episodes(
            show_name=show_name,
            season=season,
            tmdb_api_key=args.tmdb_key,
            episodes_file=episodes_file_path,
            display=display
        )
        
        if display:
            display.add_log(f"Total episode titles: {len(episode_titles)}")
        
        matches = process_video_files(
            input_directory, 
            episode_titles, 
            ocr_config, 
            display, 
            episode_info=episode_info, 
            show_name=show_name
        )
        
        if display and display.early_exit_requested:
            display.add_log("Early exit requested - showing preview with current matches")
            
        rename_preview = generate_rename_preview(matches, show_name, season, episode_info)
        display_rename_preview(rename_preview, display, show_name, season)
        
        if not args.dry_run:
            execute_renames(rename_preview, input_directory)
            
    except KeyboardInterrupt:
        if display:
            display.stop()
        sys.exit(130)
    finally:
        if display:
            display.stop()
        if args.profile:
            profiler.save_results()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

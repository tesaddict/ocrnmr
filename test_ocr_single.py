#!/usr/bin/env python3
"""
Quick test script for OCR parameter tuning with a single file.

This script allows rapid iteration on OCR parameters by:
1. Loading settings from ocr_config.json (optional)
2. Testing a single file with expected match
3. Showing detailed diagnostic information
4. Providing clear pass/fail results

Usage:
    python3 test_ocr_single.py
    python3 test_ocr_single.py --file /path/to/episode.mkv
    python3 test_ocr_single.py --config ocr_config.json --file /path/to/episode.mkv

To adjust parameters:
    1. Create ocr_config.json (optional - defaults will be used if not present)
    2. Edit ocr_config.json with OCR settings
    3. Run the script again
    4. Repeat until you get consistent matches

The script will show:
    - Current configuration
    - Frame extraction statistics
    - Match results
    - Diagnostic information if no match found
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ocrnmr import find_episode_by_title_card
from ocrnmr import extract_text_from_frame, match_episode, match_episode_with_scores
from ocrnmr.processor import FrameCache
from ocrnmr.episode_fetcher import fetch_episodes, load_episodes_file

console = Console()

# Config will be loaded in main() - these are placeholders
CONFIG = {}
OCR_CONFIG = {}
DEBUG_CONFIG = {}
TEST_CONFIG = {}

# Set up logging based on config
log_level = getattr(logging, DEBUG_CONFIG.get("log_level", "INFO").upper(), logging.INFO)

# Logging will be configured in main()


def print_config(ocr_config: Dict):
    """Print current configuration."""
    
    config_lines = []
    config_lines.append("Current Configuration:")
    config_lines.append("")
    table = Table(show_header=True, header_style="bold yellow")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Max Dimension", str(ocr_config.get("max_dimension")) if ocr_config.get("max_dimension") else "Full resolution")
    duration = ocr_config.get("duration")
    table.add_row("Duration Limit", f"{duration}s ({duration/60:.1f} min)" if duration else "Full video")
    table.add_row("Frame Interval", str(ocr_config.get("frame_interval", 2.0)))
    table.add_row("Match Threshold", str(ocr_config.get("match_threshold", 0.6)))
    
    console.print(table)
    console.print()


def print_frame_stats(video_file: Path, ocr_config: Dict):
    """Print frame extraction statistics."""
    if not DEBUG_CONFIG.get("show_frame_counts", False):
        return
    
    console.print("[bold cyan]Frame Extraction Statistics:[/bold cyan]")
    
    try:
        import ffmpeg
        probe = ffmpeg.probe(str(video_file))
        video_duration = float(probe['format'].get('duration', 0))
        console.print(f"  Video duration: {video_duration:.1f}s ({video_duration/60:.1f} min)")
        
        max_dimension = ocr_config.get("max_dimension")
        duration = ocr_config.get("duration")
        frame_interval = ocr_config.get("frame_interval", 2.0)
        
        effective_duration = duration if duration else video_duration
        estimated_frames = int(effective_duration / frame_interval)
        console.print(f"  Estimated frames ({frame_interval}s interval): ~{estimated_frames}")
        
        console.print()
    except Exception as e:
        console.print(f"[yellow]  Could not calculate frame statistics: {e}[/yellow]\n")


def test_single_file(test_file_config: dict, ocr_config: Dict, test_index: int = 0, total_tests: int = 1):
    """
    Test OCR matching on a single file.
    
    Args:
        test_file_config: Dictionary with 'test_file' and optionally 'expected_match' keys
        test_index: Index of current test (0-based)
        total_tests: Total number of tests being run
    
    Returns:
        Tuple of (success: bool, matched_episode: Optional[str])
    """
    test_file_path = test_file_config.get("test_file", "")
    expected_match = test_file_config.get("expected_match")  # Can be None
    
    # Expand ~ in path (expanduser() handles both ~ and regular paths)
    test_file = Path(test_file_path).expanduser()
    
    if not test_file.exists():
        error_msg = f"Test file not found: {test_file}"
        console.print(f"[red]✗ {error_msg}[/red]")
        console.print("[yellow]Please update test_file path in ocr_config.json[/yellow]")
        return (False, None)
    
    # Show test header if multiple tests
    if total_tests > 1:
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold cyan]Test {test_index + 1} of {total_tests}[/bold cyan]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")
    
    console.print(f"[bold]Test File:[/bold] {test_file.name}")
    if expected_match:
        console.print(f"[bold]Expected Match:[/bold] {expected_match}")
    else:
        console.print(f"[bold]Expected Match:[/bold] [dim]Not specified[/dim]")
    console.print(f"[bold]Episode Titles (for matching):[/bold] {len(EPISODE_TITLES)} titles")
    console.print()
    
    # Print configuration (only on first test)
    if test_index == 0:
        print_config(ocr_config)
    
    # Print frame statistics
    print_frame_stats(test_file, ocr_config)
    
    # Get OCR parameters from CLI args (already processed in main())
    max_dimension = ocr_config.get("max_dimension")
    duration = test_file_config.get("duration")  # Use duration from test_file_config if available
    if duration is None:
        duration = ocr_config.get("duration")
    frame_interval = ocr_config.get("frame_interval", 2.0)
    match_threshold = ocr_config.get("match_threshold", 0.6)
    enable_profiling = True  # Profiling enabled by default
    hwaccel = ocr_config.get("hwaccel")
    show_all_text = test_file_config.get("show_all_text", False)
    
    # Show extraction progress if enabled
    if DEBUG_CONFIG.get("show_extraction_progress", False):
        import ffmpeg
        try:
            probe = ffmpeg.probe(str(test_file))
            video_duration = float(probe['format'].get('duration', 0))
            console.print(f"[dim]Processing video ({video_duration:.1f}s)...[/dim]")
            if duration:
                console.print(f"[dim]Limiting to first {duration:.1f}s ({duration/60:.1f} min)[/dim]")
        except Exception:
            pass
        console.print()
    
    # Run OCR matching
    console.print("[bold yellow]Running OCR matching...[/bold yellow]\n")
    
    # Get pre-extracted frames from cache if available
    # Note: frame_cache is passed via global variable set in main
    pre_extracted_frames = None
    if 'frame_cache' in globals() and globals()['frame_cache'] is not None:
        frame_cache = globals()['frame_cache']
        # Check if extraction was started for this file
        with frame_cache.lock:
            extraction_started = test_file in frame_cache.cache
        
        if extraction_started:
            console.print(f"[dim]Retrieving frames from cache...[/dim]")
            pre_extracted_frames = frame_cache.get_frames(test_file)
            if pre_extracted_frames:
                console.print(f"[green]✓ Got {len(pre_extracted_frames)} pre-extracted frames from cache[/green]")
            else:
                console.print(f"[yellow]⚠ Extraction failed or not ready, will extract synchronously[/yellow]")
        else:
            console.print(f"[dim]No extraction started for this file, extracting frames synchronously...[/dim]")
    else:
        console.print(f"[dim]No frame cache available, extracting frames synchronously...[/dim]")
    
    # Get start_time for extraction if specified
    start_time = test_file_config.get("start_time")
    
    try:
        # If show_all_text is enabled, extract frames and show all OCR text
        if show_all_text:
            console.print("[bold cyan]Extracting frames and showing all OCR text...[/bold cyan]\n")
            from ocrnmr.frame_extractor import extract_frames_batch
            
            frames = pre_extracted_frames
            if frames is None:
                frames = extract_frames_batch(
                    test_file,
                    interval_seconds=frame_interval,
                    max_dimension=max_dimension,
                    duration=duration,
                    hwaccel=hwaccel,
                    start_time=start_time
                )
            
            console.print(f"[bold]Found {len(frames)} frames[/bold]\n")
            console.print("[bold]OCR Text from all frames with matching scores:[/bold]\n")
            
            # Sort scores by value for display
            from rich.table import Table as RichTable
            
            for timestamp, frame_bytes in frames:
                absolute_time = timestamp + (start_time if start_time else 0)
                text = extract_text_from_frame(frame_bytes)
                if text:
                    match, scores = match_episode_with_scores(text, EPISODE_TITLES, threshold=match_threshold)
                    
                    # Format timestamp
                    time_str = f"{int(absolute_time//60)}:{int(absolute_time%60):02d}"
                    
                    # Show OCR text
                    console.print(f"\n[bold cyan][{time_str}] ({absolute_time:6.1f}s)[/bold cyan]")
                    console.print(f"  OCR Text: {text[:200]}")
                    
                    # Show match result
                    if match:
                        console.print(f"  [green]✓ MATCH: {match}[/green] (threshold: {match_threshold})")
                    else:
                        console.print(f"  [yellow]✗ No match (threshold: {match_threshold})[/yellow]")
                    
                    # Show top scores
                    if scores:
                        # Sort by score descending
                        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                        # Show top 5 scores
                        top_scores = sorted_scores[:5]
                        
                        score_table = RichTable(show_header=True, header_style="bold yellow", box=None, padding=(0, 2))
                        score_table.add_column("Episode", style="cyan", overflow="fold")
                        score_table.add_column("Score", style="green", justify="right")
                        score_table.add_column("Status", style="dim", width=10)
                        
                        for ep_name, score in top_scores:
                            if score >= match_threshold:
                                status = "[green]MATCH[/green]"
                            elif score >= match_threshold * 0.8:
                                status = "[yellow]close[/yellow]"
                            else:
                                status = "[dim]low[/dim]"
                            score_table.add_row(ep_name, f"{score:.3f}", status)
                        
                        console.print("  Top matches:")
                        console.print(score_table)
                else:
                    absolute_time = timestamp + (start_time if start_time else 0)
                    time_str = f"{int(absolute_time//60)}:{int(absolute_time%60):02d}"
                    console.print(f"  [{time_str}] ({absolute_time:6.1f}s) [dim](no text detected)[/dim]")
            
            console.print()
            # Continue with normal matching after showing all text
            # IMPORTANT: Clear any cached frames so the two-pass matcher re-extracts and
            # controls its own downscaled/full-res retries.
            pre_extracted_frames = None 
        
        result = find_episode_by_title_card(
            test_file,
            EPISODE_TITLES,
            match_threshold=match_threshold,
            return_details=True,
            duration=duration,
            frame_interval=frame_interval,
            enable_profiling=enable_profiling,
            hwaccel=hwaccel,
            pre_extracted_frames=pre_extracted_frames,
            start_time=start_time
        )
        
        # Display results
        
        if result:
            matched_file, matched_episode, timestamp, extracted_text = result
            
            # Check if match is expected
            is_expected = False
            if expected_match:
                # Check if matched episode matches the expected match (exact or fuzzy)
                is_expected = (matched_episode == expected_match or 
                              matched_episode in EPISODE_TITLES and expected_match in EPISODE_TITLES)
            
            # Print results to console
            console.print("\n[bold cyan]Results:[/bold cyan]")
            if expected_match and is_expected:
                console.print(f"[bold green]✓ MATCH FOUND (Expected)[/bold green]")
            elif expected_match:
                console.print(f"[bold yellow]⚠ MATCH FOUND (Unexpected)[/bold yellow]")
            else:
                console.print(f"[bold green]✓ MATCH FOUND[/bold green]")
                console.print(f"  [dim]Matched episode from episode_titles list[/dim]")
            
            console.print(f"  Episode: {matched_episode}")
            console.print(f"  Timestamp: {timestamp:.1f}s ({int(timestamp//60)}:{int(timestamp%60):02d})")
            
            if DEBUG_CONFIG.get("show_ocr_results", False) and extracted_text:
                console.print(f"\n  [dim]Extracted Text:[/dim]")
                console.print(f"  [dim]{extracted_text[:200]}...[/dim]")
            
            if expected_match:
                console.print(f"\n  Expected: {expected_match}")
            if not is_expected:
                console.print(f"  [yellow]Got: {matched_episode}[/yellow]")
            
            return (is_expected if expected_match else True, matched_episode)
        else:
            console.print("[bold red]✗ NO MATCH FOUND[/bold red]")
            if expected_match:
                console.print(f"  Expected: {expected_match}")
            
            # If debug enabled, show some diagnostic info
            if DEBUG_CONFIG.get("show_match_attempts", False):
                console.print("\n  [dim]Attempting to find why no match...[/dim]")
                # Try extracting a few frames manually to see what OCR gets
                try:
                    from ocrnmr.frame_extractor import extract_frames_batch
                    
                    console.print("  [dim]Extracting sample frames...[/dim]")
                    start_time = test_file_config.get("start_time")
                    frames = extract_frames_batch(
                        test_file,
                        interval_seconds=frame_interval,
                        max_dimension=max_dimension,
                        duration=duration,
                        start_time=start_time
                    )
                    frame_gen = iter(frames)
                    
                    sample_count = 0
                    for timestamp, frame_bytes in frame_gen:
                        if sample_count >= 5:  # Show first 5 frames
                            break
                        text = extract_text_from_frame(frame_bytes)
                        if text:
                            frame_msg = f"Frame at {timestamp:.1f}s: {text[:100]}..."
                            console.print(f"    Frame at {timestamp:.1f}s: [dim]{text[:100]}...[/dim]")
                            match = match_episode(text, EPISODE_TITLES, threshold=match_threshold)
                            if match:
                                console.print(f"      → [green]Would match: {match}[/green]")
                            else:
                                console.print(f"      → [yellow]No match (best score below {match_threshold})[/yellow]")
                        sample_count += 1
                except Exception as e:
                    error_msg = f"Could not extract sample frames: {e}"
                    console.print(f"  [yellow]{error_msg}[/yellow]")
            
            return (False, None)
            
    except Exception as e:
        error_msg = f"ERROR: {e}"
        console.print(f"[bold red]✗ ERROR:[/bold red] {e}")
        import traceback
        if DEBUG_CONFIG.get("enabled", False):
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return (False, None)


def main():
    """Main entry point for OCR test script."""
    global CONFIG, DEBUG_CONFIG, TEST_CONFIG, EPISODE_TITLES, test_files_config
    
    # Parse command-line arguments - similar to main CLI but with debug options
    parser = argparse.ArgumentParser(
        description="Test OCR parameter tuning with a single file or directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test single file with TMDB:
  python3 test_ocr_single.py --show "Star Trek: TNG" --season 1 --file "/path/to/episode.mkv"
  
  # Test with custom episodes file:
  python3 test_ocr_single.py --show "Dragon Ball Z Kai" --season 5 --file "/path/to/episode.mkv" --episodes-file episode_config.json
  
  # Test specific time range:
  python3 test_ocr_single.py --show "The Office" --season 1 --file "/path/to/episode.mkv" --start-time 300 --end-time 320
  
  # Show all OCR text for debugging:
  python3 test_ocr_single.py --show "Stargate SG-1" --season 2 --file "/path/to/episode.mkv" --show-all-text
  
  # Override OCR settings:
  python3 test_ocr_single.py --show "The Simpsons" --season 1 --file "/path/to/episode.mkv" --max-dimension 1200 --match-threshold 0.5
        """
    )
    
    # Required arguments (same as main CLI)
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
        "--file",
        type=str,
        required=True,
        help="Path to a single video file to test (required)"
    )
    
    # Optional arguments (same as main CLI)
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
    
    # OCR settings (same as main CLI, all optional with defaults)
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
    
    # Debug/test-specific arguments
    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Start time in seconds for frame extraction (debug option, default: 0)"
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="End time in seconds for frame extraction (debug option, overrides duration if set)"
    )
    parser.add_argument(
        "--show-all-text",
        action="store_true",
        help="Show all OCR text extracted from every frame (debug option)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to ocr_config.json for additional debug/test settings (optional)"
    )
    
    args = parser.parse_args()
    
    # Load optional debug config file
    if args.config:
        CONFIG_FILE = Path(args.config).expanduser()
    else:
        CONFIG_FILE = Path("ocr_config.json")
        if not CONFIG_FILE.exists():
            CONFIG_FILE = Path(__file__).parent / "ocr_config.json"
    
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                CONFIG = json.load(f)
            console.print(f"[green]✓ Loaded debug config from: {CONFIG_FILE}[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠ Could not load debug config file: {e}[/yellow]")
            CONFIG = {}
    else:
        CONFIG = {}
    
    DEBUG_CONFIG = CONFIG.get("debug", {})
    TEST_CONFIG = CONFIG.get("test", {})
    
    # Set up logging based on debug config
    log_level = getattr(logging, DEBUG_CONFIG.get("log_level", "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Validate test file
    test_file = Path(args.file).expanduser()
    if not test_file.exists():
        console.print(f"[red]✗ Test file not found: {test_file}[/red]")
        return 1
    
    # Calculate duration if end_time is specified
    duration = args.duration if args.duration > 0 else None
    if args.end_time is not None and args.start_time is not None:
        duration = args.end_time - args.start_time
    elif args.end_time is not None:
        duration = args.end_time
    
    # Build OCR config from CLI arguments (same as main CLI)
    ocr_config = {
        "duration": duration,
        "frame_interval": args.frame_interval,
        "match_threshold": args.match_threshold,
        "hwaccel": args.hwaccel,
    }
    
    # Validate episodes file exists if provided
    if args.episodes_file:
        episodes_path = Path(args.episodes_file).expanduser()
        if not episodes_path.exists():
            console.print(f"[red]Error:[/red] Episodes file not found: {episodes_path}")
            return 1
    
    # Prepare test file config
    test_files_config = [{
        "test_file": str(test_file),
        "start_time": args.start_time,
        "end_time": args.end_time,
        "duration": duration,
        "show_all_text": args.show_all_text,
    }]
    
    console.print(f"[cyan]Testing file: {test_file.name}[/cyan]")
    if args.start_time is not None:
        console.print(f"[cyan]Start time: {args.start_time:.1f}s[/cyan]")
    if args.end_time is not None:
        console.print(f"[cyan]End time: {args.end_time:.1f}s[/cyan]")
    if duration is not None:
        console.print(f"[cyan]Duration: {duration:.1f}s[/cyan]")
    
    # Fetch episodes from TMDB and/or episodes file (same as main CLI)
    show_name = args.show
    season = args.season
    episodes_file_path = Path(args.episodes_file).expanduser() if args.episodes_file else None
    
    try:
        EPISODE_TITLES, episode_info = fetch_episodes(
            show_name=show_name,
            season=season,
            tmdb_api_key=args.tmdb_key,
            episodes_file=episodes_file_path,
            display=None
        )
        console.print(f"[green]✓ Loaded {len(EPISODE_TITLES)} episode titles[/green]")
    except SystemExit:
        # fetch_episodes exits if no episodes found - use fallback
        console.print(f"[yellow]⚠ Could not load episodes, using fallback titles[/yellow]")
        EPISODE_TITLES = TEST_CONFIG.get("episode_titles", [
            "Seven Years Later! Starting Today, Gohan Is a High School Student",
            "7 Years Since That Event! Starting Today, Gohan's a High Schooler"
        ])
        episode_info = {}
    except Exception as e:
        console.print(f"[yellow]⚠ Error loading episodes: {e}[/yellow]")
        EPISODE_TITLES = TEST_CONFIG.get("episode_titles", [
            "Seven Years Later! Starting Today, Gohan Is a High School Student",
            "7 Years Since That Event! Starting Today, Gohan's a High Schooler"
        ])
        episode_info = {}
    
    total_tests = len(test_files_config)
    results = []
    matches = []  # Store (file_path, matched_episode) tuples
    
    console.print("[bold]OCR Parameter Tuning Test[/bold]")
    console.print("=" * 60)
    
    # Create list of file paths for index lookup
    file_paths_list = [Path(test_file_config.get("test_file", "")).expanduser() for test_file_config in test_files_config]
    
    # Create frame cache for pipelined extraction
    frame_cache = FrameCache(memory_limit_bytes=1073741824)  # 1GB limit
    # Make frame_cache available to test_single_file via globals
    globals()['frame_cache'] = frame_cache
    
    # Prepare extraction config from ocr_config
    # Get start_time from first test file config (if specified)
    start_time = None
    if test_files_config and test_files_config[0].get("start_time") is not None:
        start_time = test_files_config[0].get("start_time")
    
    extraction_config = {
        "max_dimension": ocr_config.get("max_dimension"),
        "duration": ocr_config.get("duration"),
        "frame_interval": ocr_config.get("frame_interval", 2.0),
        "hwaccel": ocr_config.get("hwaccel"),
        "start_time": start_time
    }
    
    # Start extraction for ALL files immediately - push ahead as fast as possible
    # This allows FFmpeg to finish early and free resources for EasyOCR
    if test_files_config:
        for file_idx, test_file_config in enumerate(test_files_config):
            file_path = Path(test_file_config.get("test_file", "")).expanduser()
            # Use start_time and duration from individual file config if available
            file_start_time = test_file_config.get("start_time")
            file_duration = test_file_config.get("duration")
            file_extraction_config = extraction_config.copy()
            if file_start_time is not None:
                file_extraction_config["start_time"] = file_start_time
            if file_duration is not None:
                file_extraction_config["duration"] = file_duration
            frame_cache.start_extraction(file_path, file_extraction_config)
    
    # Run all test files
    for idx, test_file_config in enumerate(test_files_config):
        test_file_path = Path(test_file_config.get("test_file", "")).expanduser()
        
        # Process current file (frames should already be ready or will wait)
        success, matched_episode = test_single_file(test_file_config, ocr_config, test_index=idx, total_tests=total_tests)
        results.append(success)
        matches.append((test_file_path, matched_episode))
        
        # Clear frames from cache after processing
        frame_cache.cleanup(test_file_path)
    
    # Shutdown frame cache executor
    frame_cache.shutdown()
    
    # Generate rename preview data using show_name and season from CLI args
    # Import filename generation function
    from ocrnmr.filename import sanitize_filename
    
    # Generate rename preview using episode_info if available
    rename_preview = []
    for file_path, matched_episode in matches:
        original_name = file_path.name
        ext = file_path.suffix
        
        if matched_episode:
            # Use episode_info if available, otherwise sequential numbering
            if matched_episode in episode_info:
                ep_season, ep_num = episode_info[matched_episode]
                new_name = f"{show_name} - S{ep_season:02d}E{ep_num:02d} - {matched_episode}{ext}"
            else:
                # Fallback to sequential numbering
                unique_episodes = set(ep for _, ep, _ in rename_preview if ep)
                ep_num = len(unique_episodes) + 1
                new_name = f"{show_name} - S{season:02d}E{ep_num:02d} - {matched_episode}{ext}"
            
            new_name = sanitize_filename(new_name)
            rename_preview.append((original_name, new_name, matched_episode))
        else:
            rename_preview.append((original_name, None, None))
    
    matched_count = sum(1 for _, new_name, _ in rename_preview if new_name)
    
    # Show rename preview in console
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]Test Summary[/bold cyan]")
    console.print("=" * 60)
    
    passed = sum(results)
    failed = len(results) - passed
    
    for idx, (test_file_config, success) in enumerate(zip(test_files_config, results)):
        test_file = Path(test_file_config.get("test_file", "")).expanduser().name
        status = "[green]✓ PASSED[/green]" if success else "[red]✗ FAILED[/red]"
        console.print(f"  Test {idx + 1}: {test_file} - {status}")
    
    console.print(f"\n[bold]Total: {passed} passed, {failed} failed out of {total_tests} tests[/bold]")
    
    # Show rename preview
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]Rename Preview[/bold cyan]")
    console.print("=" * 60)
    
    from rich.table import Table
    preview_table = Table(show_header=True, header_style="bold yellow")
    preview_table.add_column("Original", style="cyan", overflow="fold")
    preview_table.add_column("→", style="dim", width=3, justify="center")
    preview_table.add_column("New Name", style="green", overflow="fold")
    preview_table.add_column("Episode", style="dim", overflow="fold")
    
    for original, new_name, episode in rename_preview:
        if new_name:
            preview_table.add_row(original, "→", new_name, episode or "")
        else:
            preview_table.add_row(original, "→", "[red]No match found[/red]", "")
    
    console.print()
    console.print(preview_table)
    console.print(f"\n[dim]Matched: {matched_count}/{len(rename_preview)} files[/dim]")
    
    if all(results):
        console.print("\n[bold green]✓ All tests PASSED[/bold green]")
        console.print("\n[yellow]Tip:[/yellow] Adjust parameters in ocr_config.json and run again to optimize")
        return 0
    else:
        console.print("\n[bold red]✗ Some tests FAILED[/bold red]")
        console.print("\n[yellow]Tip:[/yellow] Try adjusting:")
        console.print("  - duration (limit to first N seconds)")
        console.print("  - max_dimension (null for full resolution)")
        console.print("  - frame_interval (lower = more frames, default: 1.0-2.0)")
        console.print("  - match_threshold (lower = more lenient matching)")
        return 1


if __name__ == "__main__":
    sys.exit(main())


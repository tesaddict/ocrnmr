#!/usr/bin/env python3
"""
Quick test script for OCR parameter tuning with a single file.

This script allows rapid iteration on OCR parameters by:
1. Loading all settings from ocr_config.json
2. Testing a single file with expected match
3. Showing detailed diagnostic information
4. Providing clear pass/fail results

Usage:
    python3 test_ocr_single.py

To adjust parameters:
    1. Edit ocr_config.json
    2. Run the script again
    3. Repeat until you get consistent matches

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
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, Future
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ocrnmr.ocr import find_episode_by_title_card
from ocrnmr.ocr import extract_text_from_frame, match_episode, match_episode_with_scores
from ocrnmr.cli import get_episode_titles

console = Console()

# Load configuration from ocr_config.json
CONFIG_FILE = Path(__file__).parent / "ocr_config.json"
CONFIG = {}
if CONFIG_FILE.exists():
    try:
        with open(CONFIG_FILE, 'r') as f:
            CONFIG = json.load(f)
        console.print(f"[green]✓ Loaded config from: {CONFIG_FILE}[/green]")
    except Exception as e:
        console.print(f"[red]✗ Could not load config file: {e}[/red]")
        sys.exit(1)
else:
    console.print(f"[red]✗ Config file not found: {CONFIG_FILE}[/red]")
    console.print("[yellow]Please create ocr_config.json first[/yellow]")
    sys.exit(1)

# Get OCR config with defaults
OCR_CONFIG = CONFIG.get("ocr", {})
DEBUG_CONFIG = CONFIG.get("debug", {})
TEST_CONFIG = CONFIG.get("test", {})

# Set up logging based on config
log_level = getattr(logging, DEBUG_CONFIG.get("log_level", "INFO").upper(), logging.INFO)

# Logging will be configured in main()


def print_config():
    """Print current configuration."""
    
    config_lines = []
    config_lines.append("Current Configuration:")
    config_lines.append("")
    table = Table(show_header=True, header_style="bold yellow")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Max Dimension", str(OCR_CONFIG.get("max_dimension")) if OCR_CONFIG.get("max_dimension") else "Full resolution")
    duration = OCR_CONFIG.get("duration")
    table.add_row("Duration Limit", f"{duration}s ({duration/60:.1f} min)" if duration else "Full video")
    table.add_row("Frame Interval", str(OCR_CONFIG.get("frame_interval", 1.0)))
    table.add_row("Match Threshold", str(OCR_CONFIG.get("match_threshold", 0.6)))
    
    console.print(table)
    console.print()


def print_frame_stats(video_file: Path):
    """Print frame extraction statistics."""
    if not DEBUG_CONFIG.get("show_frame_counts", False):
        return
    
    console.print("[bold cyan]Frame Extraction Statistics:[/bold cyan]")
    
    try:
        import ffmpeg
        probe = ffmpeg.probe(str(video_file))
        video_duration = float(probe['format'].get('duration', 0))
        console.print(f"  Video duration: {video_duration:.1f}s ({video_duration/60:.1f} min)")
        
        max_dimension = OCR_CONFIG.get("max_dimension")
        if max_dimension is not None and max_dimension == 0:
            max_dimension = None
        duration = OCR_CONFIG.get("duration")
        if duration is not None and duration == 0:
            duration = None
        frame_interval = OCR_CONFIG.get("frame_interval", 1.0)
        
        effective_duration = duration if duration else video_duration
        estimated_frames = int(effective_duration / frame_interval)
        console.print(f"  Estimated frames ({frame_interval}s interval): ~{estimated_frames}")
        
        console.print()
    except Exception as e:
        console.print(f"[yellow]  Could not calculate frame statistics: {e}[/yellow]\n")


class FrameCache:
    """Manages pipelined FFMPEG frame extraction with memory limits."""
    
    def __init__(self, memory_limit_bytes: int = 1073741824, file_paths: Optional[List[Path]] = None):  # 1GB default
        self.memory_limit = memory_limit_bytes
        self.cache: Dict[Path, Tuple[Future, int]] = {}  # {file_path: (future, memory_bytes)}
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        self.file_paths = file_paths  # List of all file paths in order (for index lookup)
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
    
    def start_extraction(self, file_path: Path, config: Dict[str, Any]) -> None:
        """Start background frame extraction for a file."""
        with self.lock:
            if file_path in self.cache:
                # Already extracting or cached
                return
            
            # Check if we have room (we'll check again when extraction completes)
            future = self.executor.submit(self._extract_frames, file_path, config)
            # We don't know memory yet, so estimate conservatively
            self.cache[file_path] = (future, 0)
            self.logger.info(f"Started background FFMPEG extraction for {file_path.name}")
            print(f"[INFO] Started background FFMPEG extraction for {file_path.name}", file=sys.stderr)
    
    def _extract_frames(self, file_path: Path, config: Dict[str, Any]) -> List[Tuple[float, bytes]]:
        """Extract frames in background thread."""
        from ocrnmr.ocr.frame_extractor import extract_frames_batch
        import sys
        import time
        
        start_time = time.time()
        print(f"[INFO] Background extraction starting for {file_path.name}...", file=sys.stderr)
        
        try:
            # Regular batch extraction
            print(f"[INFO] Calling extract_frames_batch for {file_path.name}...", file=sys.stderr)
            frames = extract_frames_batch(
                file_path,
                interval_seconds=config.get("frame_interval", 2.0),
                max_dimension=config.get("max_dimension"),
                duration=config.get("duration"),
                hwaccel=config.get("hwaccel"),
                start_time=config.get("start_time")
            )
            extraction_time = time.time() - start_time
            print(f"[INFO] extract_frames_batch completed in {extraction_time:.2f}s for {file_path.name}", file=sys.stderr)
            
            # Update memory estimate
            print(f"[INFO] Estimating memory for {file_path.name}...", file=sys.stderr)
            memory_bytes = self.estimate_memory(frames)
            print(f"[INFO] Memory estimated: {memory_bytes / 1024 / 1024:.1f}MB for {file_path.name}", file=sys.stderr)
            
            print(f"[INFO] Acquiring lock to update cache for {file_path.name}...", file=sys.stderr)
            with self.lock:
                print(f"[INFO] Lock acquired for {file_path.name}", file=sys.stderr)
                if file_path in self.cache:
                    self.cache[file_path] = (self.cache[file_path][0], memory_bytes)
                    # Calculate total memory directly without calling get_total_memory() to avoid deadlock
                    # (we're already holding the lock)
                    total_memory = sum(mem_bytes for _, mem_bytes in self.cache.values())
                    log_msg = (
                        f"FFMPEG extraction completed for {file_path.name}: "
                        f"{len(frames)} frames, ~{memory_bytes / 1024 / 1024:.1f}MB "
                        f"(total cache: ~{total_memory / 1024 / 1024:.1f}MB)"
                    )
                    self.logger.info(log_msg)
                    print(f"[INFO] {log_msg}", file=sys.stderr)
                else:
                    print(f"[WARNING] File {file_path.name} not in cache when updating memory!", file=sys.stderr)
            
            print(f"[INFO] Lock released for {file_path.name}", file=sys.stderr)
            
            total_time = time.time() - start_time
            print(f"[INFO] Background extraction completed successfully for {file_path.name} in {total_time:.2f}s, returning {len(frames)} frames", file=sys.stderr)
            return frames
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"Background extraction failed for {file_path.name} after {elapsed:.2f}s: {e}"
            self.logger.error(error_msg)
            print(f"[ERROR] {error_msg}", file=sys.stderr)
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}", file=sys.stderr)
            with self.lock:
                if file_path in self.cache:
                    del self.cache[file_path]
            raise
    
    def get_frames(self, file_path: Path) -> Optional[List[Tuple[float, bytes]]]:
        """Get frames for a file, waiting if extraction is in progress."""
        with self.lock:
            if file_path not in self.cache:
                return None
            
            future, memory_bytes = self.cache[file_path]
        
        # Check if extraction is already complete
        if future.done():
            # Extraction already complete, get result immediately
            print(f"[INFO] Extraction already done for {file_path.name}, getting result...", file=sys.stderr)
            try:
                frames = future.result()
                print(f"[INFO] Got {len(frames) if frames else 0} frames from completed extraction for {file_path.name}", file=sys.stderr)
                return frames
            except Exception as e:
                error_msg = f"Error getting frames for {file_path.name}: {e}"
                self.logger.error(error_msg)
                print(f"[ERROR] {error_msg}", file=sys.stderr)
                import traceback
                print(f"[ERROR] Traceback: {traceback.format_exc()}", file=sys.stderr)
                with self.lock:
                    if file_path in self.cache:
                        del self.cache[file_path]
                return None
        
        # Extraction still in progress - wait for it
        print(f"[INFO] Waiting for background FFMPEG extraction to complete for {file_path.name}...", file=sys.stderr)
        
        try:
            print(f"[INFO] Calling future.result() for {file_path.name}...", file=sys.stderr)
            frames = future.result()  # This will block until extraction completes
            print(f"[INFO] future.result() returned for {file_path.name}, got {len(frames) if frames else 0} frames", file=sys.stderr)
            
            # Check memory limit after extraction completes
            print(f"[INFO] Checking memory limit for {file_path.name}...", file=sys.stderr)
            try:
                total_memory = self.get_total_memory()
                print(f"[INFO] Total memory: {total_memory / 1024 / 1024:.1f}MB", file=sys.stderr)
                if total_memory > self.memory_limit:
                    self.logger.warning(
                        f"Memory limit exceeded: {total_memory / 1024 / 1024:.1f}MB > "
                        f"{self.memory_limit / 1024 / 1024:.1f}MB. "
                        f"Consider reducing duration or frame_interval."
                    )
            except Exception as e:
                print(f"[WARNING] Error checking memory limit: {e}", file=sys.stderr)
            
            print(f"[INFO] Background extraction completed for {file_path.name}, got {len(frames)} frames, returning", file=sys.stderr)
            return frames
        except Exception as e:
            self.logger.error(f"Error getting frames for {file_path}: {e}")
            print(f"[ERROR] Background extraction failed for {file_path.name}: {e}", file=sys.stderr)
            with self.lock:
                if file_path in self.cache:
                    del self.cache[file_path]
            return None
    
    def clear(self, file_path: Path) -> None:
        """Remove frames from cache after use."""
        print(f"[INFO] clear() called for {file_path.name}", file=sys.stderr)
        with self.lock:
            print(f"[INFO] clear() acquired lock for {file_path.name}", file=sys.stderr)
            if file_path in self.cache:
                memory_bytes = self.cache[file_path][1]
                del self.cache[file_path]
                # Calculate total memory directly (we already hold the lock)
                total_memory = sum(mem_bytes for _, mem_bytes in self.cache.values())
                log_msg = (
                    f"Cleared cache for {file_path.name} (~{memory_bytes / 1024 / 1024:.1f}MB). "
                    f"Remaining cache: ~{total_memory / 1024 / 1024:.1f}MB"
                )
                self.logger.info(log_msg)
                print(f"[INFO] {log_msg}", file=sys.stderr)
            else:
                print(f"[WARNING] File {file_path.name} not in cache when trying to clear", file=sys.stderr)
        print(f"[INFO] clear() released lock for {file_path.name}", file=sys.stderr)
    
    def shutdown(self) -> None:
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)


def test_single_file(test_file_config: dict, test_index: int = 0, total_tests: int = 1):
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
        print_config()
    
    # Print frame statistics
    print_frame_stats(test_file)
    
    # Get OCR parameters from config, with overrides from test_file_config
    max_dimension = OCR_CONFIG.get("max_dimension")
    if max_dimension is not None and max_dimension == 0:
        max_dimension = None
    duration = test_file_config.get("duration")  # Use duration from test_file_config if available
    if duration is None:
        duration = OCR_CONFIG.get("duration")
    if duration is not None and duration == 0:
        duration = None
    frame_interval = test_file_config.get("frame_interval")
    if frame_interval is None:
        frame_interval = OCR_CONFIG.get("frame_interval", 1.0)
    match_threshold = OCR_CONFIG.get("match_threshold", 0.6)
    enable_profiling = True  # Profiling enabled by default
    hwaccel = OCR_CONFIG.get("hwaccel")  # Hardware acceleration (can be null)
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
            from ocrnmr.ocr.frame_extractor import extract_frames_batch
            
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
            pre_extracted_frames = frames
        
        result = find_episode_by_title_card(
            test_file,
            EPISODE_TITLES,
            match_threshold=match_threshold,
            return_details=True,
            max_dimension=max_dimension,
            duration=duration,
            frame_interval=frame_interval,
            enable_profiling=enable_profiling,
            hwaccel=hwaccel,
            pre_extracted_frames=pre_extracted_frames
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
                    from ocrnmr.ocr.frame_extractor import extract_frames_batch
                    
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
    """Main entry point for ocrnmr application."""
    global CONFIG, OCR_CONFIG, DEBUG_CONFIG, TEST_CONFIG, EPISODE_TITLES, test_files_config
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Test OCR parameter tuning with a single file or directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 test_ocr_single.py
  python3 test_ocr_single.py --config stargate.json --file "/path/to/episode.mkv"
  python3 test_ocr_single.py --config stargate.json --file "/path/to/episode.mkv" --start-time 300
  python3 test_ocr_single.py --config stargate.json --file "/path/to/episode.mkv" --start-time 310 --end-time 320
  python3 test_ocr_single.py --config stargate.json --file "/path/to/episode.mkv" --start-time 310 --end-time 320 --show-all-text
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: ocr_config.json)"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a single file to test (overrides test_directory)"
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Start time in seconds for frame extraction (default: 0)"
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="End time in seconds for frame extraction (overrides duration if set)"
    )
    parser.add_argument(
        "--show-all-text",
        action="store_true",
        help="Show all OCR text extracted from every frame"
    )
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=None,
        help="Override frame interval in seconds (default: from config)"
    )
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        CONFIG_FILE = Path(args.config).expanduser()
    else:
        CONFIG_FILE = Path("ocr_config.json")
        if not CONFIG_FILE.exists():
            CONFIG_FILE = Path(__file__).parent / "ocr_config.json"
    
    if not CONFIG_FILE.exists():
        console.print(f"[red]✗ Config file not found: {CONFIG_FILE}[/red]")
        console.print("[yellow]Please create ocr_config.json or specify --config[/yellow]")
        return 1
    
    try:
        with open(CONFIG_FILE, 'r') as f:
            CONFIG = json.load(f)
        console.print(f"[green]✓ Loaded config from: {CONFIG_FILE}[/green]")
    except Exception as e:
        console.print(f"[red]✗ Could not load config file: {e}[/red]")
        return 1
    
    OCR_CONFIG = CONFIG.get("ocr", {})
    DEBUG_CONFIG = CONFIG.get("debug", {})
    TEST_CONFIG = CONFIG.get("test", {})
    
    # Set up logging based on config
    log_level = getattr(logging, DEBUG_CONFIG.get("log_level", "INFO").upper(), logging.INFO)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test configuration from command-line or config file
    test_files_config = []
    
    if args.file:
        # Single file specified via command-line
        test_file = Path(args.file).expanduser()
        if not test_file.exists():
            console.print(f"[red]✗ Test file not found: {test_file}[/red]")
            return 1
        # Calculate duration if end_time is specified
        duration = None
        if args.end_time is not None and args.start_time is not None:
            duration = args.end_time - args.start_time
        elif args.end_time is not None:
            duration = args.end_time
        
        test_files_config.append({
            "test_file": str(test_file), 
            "start_time": args.start_time,
            "end_time": args.end_time,
            "duration": duration,
            "show_all_text": args.show_all_text,
            "frame_interval": args.frame_interval
        })
        console.print(f"[cyan]Testing single file: {test_file.name}[/cyan]")
        if args.start_time is not None:
            console.print(f"[cyan]Start time: {args.start_time:.1f}s[/cyan]")
        if args.end_time is not None:
            console.print(f"[cyan]End time: {args.end_time:.1f}s[/cyan]")
        if duration is not None:
            console.print(f"[cyan]Duration: {duration:.1f}s[/cyan]")
    else:
        # Use directory-based logic from config
        test_directory = TEST_CONFIG.get("test_directory")
        
        if test_directory:
            test_dir = Path(test_directory).expanduser()
            if test_dir.exists() and test_dir.is_dir():
                mkv_files = sorted(list(test_dir.glob("*.mkv")))
                if mkv_files:
                    for mkv_file in mkv_files:
                        duration = None
                        if args.end_time is not None and args.start_time is not None:
                            duration = args.end_time - args.start_time
                        elif args.end_time is not None:
                            duration = args.end_time
                        test_files_config.append({
                            "test_file": str(mkv_file), 
                            "start_time": args.start_time,
                            "end_time": args.end_time,
                            "duration": duration,
                            "show_all_text": args.show_all_text,
                            "frame_interval": args.frame_interval
                        })
                    console.print(f"[cyan]Found {len(mkv_files)} .mkv files in {test_dir}[/cyan]")
                else:
                    console.print(f"[yellow]No .mkv files found in {test_dir}[/yellow]")
            else:
                console.print(f"[red]Test directory not found: {test_dir}[/red]")
                return 1
        else:
            # Fallback to default directory if test_directory is not set
            default_dir = Path("~/rips/Dragon_Ball_Z_Kai/S5/").expanduser()
            if default_dir.exists() and default_dir.is_dir():
                mkv_files = sorted(list(default_dir.glob("*.mkv")))
                if mkv_files:
                    for mkv_file in mkv_files:
                        duration = None
                        if args.end_time is not None and args.start_time is not None:
                            duration = args.end_time - args.start_time
                        elif args.end_time is not None:
                            duration = args.end_time
                        test_files_config.append({
                            "test_file": str(mkv_file), 
                            "start_time": args.start_time,
                            "end_time": args.end_time,
                            "duration": duration,
                            "show_all_text": args.show_all_text,
                            "frame_interval": args.frame_interval
                        })
                    console.print(f"[cyan]Found {len(mkv_files)} .mkv files in {default_dir}[/cyan]")
                else:
                    console.print(f"[yellow]No .mkv files found in {default_dir}[/yellow]")
            else:
                console.print(f"[yellow]No test configuration found. Please set test_directory in config or use --file[/yellow]")
                test_files_config = []
    
    # Get episode titles for fuzzy matching from config's show section
    try:
        EPISODE_TITLES = get_episode_titles(CONFIG, display=None)
        console.print(f"[green]✓ Loaded {len(EPISODE_TITLES)} episode titles[/green]")
    except Exception as e:
        console.print(f"[yellow]⚠ Could not load episode titles from config: {e}[/yellow]")
        # Fallback to test config or hardcoded defaults
        EPISODE_TITLES = TEST_CONFIG.get("episode_titles", [
            "Seven Years Later! Starting Today, Gohan Is a High School Student",
            "7 Years Since That Event! Starting Today, Gohan's a High Schooler"
        ])
        console.print(f"[yellow]Using fallback episode titles: {len(EPISODE_TITLES)} titles[/yellow]")
    
    total_tests = len(test_files_config)
    results = []
    matches = []  # Store (file_path, matched_episode) tuples
    
    console.print("[bold]OCR Parameter Tuning Test[/bold]")
    console.print("=" * 60)
    
    # Create list of file paths for index lookup
    file_paths_list = [Path(test_file_config.get("test_file", "")).expanduser() for test_file_config in test_files_config]
    
    # Create frame cache for pipelined extraction
    frame_cache = FrameCache(memory_limit_bytes=1073741824, file_paths=file_paths_list)  # 1GB limit
    # Make frame_cache available to test_single_file via globals
    globals()['frame_cache'] = frame_cache
    
    # Prepare extraction config
    # Get start_time from first test file config (if specified)
    start_time = None
    if test_files_config and test_files_config[0].get("start_time") is not None:
        start_time = test_files_config[0].get("start_time")
    
    extraction_config = {
        "max_dimension": OCR_CONFIG.get("max_dimension"),
        "duration": OCR_CONFIG.get("duration"),
        "frame_interval": OCR_CONFIG.get("frame_interval", 2.0),
        "hwaccel": OCR_CONFIG.get("hwaccel"),
        "start_time": start_time
    }
    
    # Start extraction for ALL files immediately - push ahead as fast as possible
    # This allows FFMPEG to finish early and free resources for EasyOCR
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
            file_frame_interval = test_file_config.get("frame_interval")
            if file_frame_interval is not None:
                file_extraction_config["frame_interval"] = file_frame_interval
            frame_cache.start_extraction(file_path, file_extraction_config)
            print(f"[INFO] Queued FFMPEG extraction for file {file_idx + 1}/{len(test_files_config)}: {file_path.name}", file=sys.stderr)
    
    # Run all test files
    for idx, test_file_config in enumerate(test_files_config):
        test_file_path = Path(test_file_config.get("test_file", "")).expanduser()
        
        # Process current file (frames should already be ready or will wait)
        print(f"[INFO] Calling test_single_file for {test_file_path.name}...", file=sys.stderr)
        success, matched_episode = test_single_file(test_file_config, test_index=idx, total_tests=total_tests)
        print(f"[INFO] test_single_file completed for {test_file_path.name}, success={success}", file=sys.stderr)
        results.append(success)
        matches.append((test_file_path, matched_episode))
        
        # Clear frames from cache after processing
        print(f"[INFO] Clearing cache for {test_file_path.name}...", file=sys.stderr)
        frame_cache.clear(test_file_path)
        print(f"[INFO] Cache cleared for {test_file_path.name}", file=sys.stderr)
        print(f"[INFO] Completed processing file {idx + 1}/{len(test_files_config)}: {test_file_path.name}", file=sys.stderr)
        print(f"[INFO] Moving to next file in loop...", file=sys.stderr)
    
    # Shutdown frame cache executor
    frame_cache.shutdown()
    
    # Generate rename preview data
    # Try to determine show name and season from directory structure
    # Default to "Dragon Ball Z Kai" and season 5 if we can't determine
    show_name = "Dragon Ball Z Kai"
    season = 5
    
    if test_files_config:
        first_file = Path(test_files_config[0].get("test_file", "")).expanduser()
        if first_file.exists():
            # Try to extract from directory path (e.g., ~/rips/Dragon_Ball_Z_Kai/S5/)
            parts = first_file.parent.parts
            for i, part in enumerate(parts):
                if part.startswith('S') and part[1:].isdigit():
                    season = int(part[1:])
                    # Show name might be in parent directory
                    if i > 0:
                        show_name = parts[i-1].replace('_', ' ').title()
                    break
    
    # Import filename generation function
    from ocrnmr.filename import sanitize_filename
    
    # Group matches by episode to assign episode numbers
    episode_to_num = {}
    episode_num = 1
    for file_path, matched_episode in matches:
        if matched_episode and matched_episode not in episode_to_num:
            episode_to_num[matched_episode] = episode_num
            episode_num += 1
    
    # Generate rename preview
    rename_preview = []
    for idx, (file_path, matched_episode) in enumerate(matches):
        original_name = file_path.name
        ext = file_path.suffix
        
        if matched_episode:
            ep_num = episode_to_num.get(matched_episode, idx + 1)
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


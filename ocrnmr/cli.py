#!/usr/bin/env python3
"""Command-line interface for OCR NMR application."""

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

try:
    import requests
except ImportError:
    requests = None

# Add parent directory to path for direct script execution
# This allows the script to be run as: python3 ocrnmr/cli.py
script_dir = Path(__file__).parent.parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# All imports at the top - simpler and more reliable
from ocrnmr.display import OCRProgressDisplay
from ocrnmr.processor import PipelinedOCRProcessor
from ocrnmr.tmdb_client import TMDBClient
from ocrnmr.filename import sanitize_filename
from ocrnmr.exit_flag import FORCE_EXIT

# Rich imports for help formatting and preview display
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

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


class RichHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Custom help formatter using rich for beautiful output."""
    
    def __init__(self, prog):
        super().__init__(prog, max_help_position=50, width=120)
        self.console = Console()
    
    def _format_action_invocation(self, action):
        """Format the action invocation (argument name) with rich colors."""
        try:
            if not action.option_strings:
                metavar, = self._metavar_formatter(action, action.dest)(1)
                return f"[yellow]{metavar}[/yellow]"
            else:
                parts = []
                # Format option strings with cyan color
                for option_string in action.option_strings:
                    parts.append(f"[cyan]{option_string}[/cyan]")
                # Add metavar if present
                if hasattr(action, 'metavar') and action.metavar is not None:
                    parts.append(f"[yellow]{action.metavar}[/yellow]")
                return ' '.join(parts)
        except Exception:
            # Fallback to parent implementation if anything goes wrong
            return super()._format_action_invocation(action)
    
    def _format_action(self, action):
        """Format a single action with rich markup."""
        try:
            # Get the formatted help text
            help_text = self._expand_help(action) if action.help else ""
            
            # Format the action invocation
            action_header = self._format_action_invocation(action)
            
            # Mark required arguments
            if hasattr(action, 'required') and action.required:
                action_header = f"[bold]{action_header}[/bold] [red](required)[/red]"
            
            # Format help text - highlight defaults
            if help_text:
                # Replace default mentions with styled version
                import re
                help_text = re.sub(r'\(default:\s*([^)]+)\)', r'[dim](default: \1)[/dim]', help_text)
                return f"  {action_header}\n      {help_text}\n"
            else:
                return f"  {action_header}\n"
        except Exception:
            # Fallback to parent implementation if anything goes wrong
            return super()._format_action(action)
    
    def format_help(self):
        """Format the entire help message with rich markup."""
        try:
            # Get the base formatted help from parent (this includes our rich markup from _format_action)
            help_text = super().format_help()
            
            # Apply additional rich formatting to sections
            lines = help_text.split('\n')
            formatted_lines = []
            
            for line in lines:
                # Format usage line
                if line.strip().startswith('usage:'):
                    formatted_lines.append(f"[bold cyan]{line.strip()}[/bold cyan]")
                # Format section headers (lines that are all caps or have colons and aren't indented)
                elif line and not line.startswith(' ') and (line.isupper() or (':' in line and len(line.strip()) < 50)):
                    formatted_lines.append(f"[bold]{line}[/bold]")
                # Format epilog examples (comment lines)
                elif line.strip().startswith('#'):
                    formatted_lines.append(f"  [dim]{line.strip()}[/dim]")
                # Format JSON in epilog
                elif line.strip().startswith('{') or line.strip().startswith('}'):
                    formatted_lines.append(f"  [yellow]{line.strip()}[/yellow]")
                # Format command examples (lines with ocrnmr that aren't indented much)
                elif line.strip() and 'ocrnmr' in line and line.startswith('  '):
                    formatted_lines.append(f"  [green]{line.strip()}[/green]")
                else:
                    formatted_lines.append(line)
            
            return '\n'.join(formatted_lines)
        except Exception:
            # Fallback to parent implementation if anything goes wrong
            return super().format_help()


def load_config(config_path: Path) -> Dict:
    """Load configuration from JSON file."""
    console = Console(stderr=True)
    if not config_path.exists():
        console.print(f"[red]Error:[/red] Config file not found: {config_path}")
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        console.print(f"[red]Error:[/red] Invalid JSON in config file: {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error loading config file:[/red] {e}")
        sys.exit(1)


def get_episode_titles(config: Dict, display: Optional[OCRProgressDisplay] = None) -> List[str]:
    """Get episode titles from TMDB and/or config."""
    show_config = config.get("show", {})
    episode_titles = []
    
    show_name = show_config.get("show_name")
    season = show_config.get("season")
    tmdb_api_key = show_config.get("tmdb_api_key")
    
    # Support new format: episodes array with explicit episode numbers
    episodes = show_config.get("episodes", [])
    # Support old format: flat episode_titles list (backward compatibility)
    episode_titles_old = show_config.get("episode_titles", [])
    
    # Extract titles from new episodes format
    manual_titles = []
    if episodes:
        # New format: episodes array
        for ep_entry in episodes:
            if isinstance(ep_entry, dict) and "title" in ep_entry:
                manual_titles.append(ep_entry["title"])
            elif isinstance(ep_entry, str):
                # Handle case where episodes might be a list of strings (fallback)
                manual_titles.append(ep_entry)
    elif episode_titles_old:
        # Old format: flat list (backward compatibility)
        manual_titles = episode_titles_old
    
    # Track TMDB lookup status for better error messages
    tmdb_status = {
        'api_key_provided': False,
        'api_key_valid': None,  # None = not checked, True = valid, False = invalid
        'show_found': None,
        'season_found': None,
        'error_message': None
    }
    
    # Try to fetch from TMDB if show_name and season are provided
    if show_name and season:
        try:
            # Get API key from config or environment
            api_key = tmdb_api_key
            if not api_key:
                api_key = os.getenv('TMDB_API_KEY')
            
            if api_key:
                tmdb_status['api_key_provided'] = True
                if display:
                    display.add_ocr_diagnostic(f"Fetching episode titles from TMDB for {show_name} Season {season}...")
                
                try:
                    client = TMDBClient(api_key=api_key)
                    tmdb_titles = client.get_episode_titles(show_name, season)
                    
                    if tmdb_titles:
                        episode_titles.extend(tmdb_titles)
                        tmdb_status['api_key_valid'] = True
                        tmdb_status['show_found'] = True
                        tmdb_status['season_found'] = True
                        if display:
                            display.add_ocr_diagnostic(f"Found {len(tmdb_titles)} episodes from TMDB")
                    else:
                        # Check if show was found but season wasn't, or if show wasn't found
                        # This helps diagnose the issue
                        try:
                            shows = client.search_tv_show(show_name)
                            if shows:
                                tmdb_status['api_key_valid'] = True
                                tmdb_status['show_found'] = True
                                show_id = shows[0]['id']
                                season_data = client.get_season(show_id, season)
                                if season_data:
                                    tmdb_status['season_found'] = True
                                    tmdb_status['error_message'] = f"Season {season} found but has no episodes"
                                else:
                                    tmdb_status['season_found'] = False
                                    tmdb_status['error_message'] = f"Season {season} not found for this show"
                            else:
                                tmdb_status['api_key_valid'] = True
                                tmdb_status['show_found'] = False
                                tmdb_status['error_message'] = f"Show '{show_name}' not found in TMDB"
                        except Exception as diag_e:
                            # If diagnostic check fails, assume API is valid but couldn't diagnose
                            # Don't mark API as invalid just because diagnostics failed
                            tmdb_status['api_key_valid'] = True
                            tmdb_status['error_message'] = f"No episodes found (diagnostic check failed: {diag_e})"
                        
                        if display:
                            display.add_ocr_diagnostic("No episodes found in TMDB")
                except requests.exceptions.HTTPError as e:
                    # Only mark API as invalid for actual HTTP errors
                    if hasattr(e, 'response') and e.response.status_code == 401:
                        tmdb_status['api_key_valid'] = False
                        tmdb_status['error_message'] = "Invalid TMDB API key (401 Unauthorized)"
                        if display:
                            display.add_ocr_diagnostic("Error: Invalid TMDB API key")
                    else:
                        tmdb_status['api_key_valid'] = None  # Unknown - could be network issue
                        tmdb_status['error_message'] = f"TMDB API HTTP error: {e} (status {getattr(e.response, 'status_code', 'unknown')})"
                        if display:
                            display.add_ocr_diagnostic(f"Error fetching from TMDB: {e}")
                except Exception as e:
                    # For other exceptions, don't assume API is invalid
                    # It could be network issues, timeouts, etc.
                    tmdb_status['api_key_valid'] = None  # Unknown
                    tmdb_status['error_message'] = f"Error accessing TMDB: {e}"
                    if display:
                        display.add_ocr_diagnostic(f"Error fetching from TMDB: {e}")
                        display.add_ocr_diagnostic("Continuing with manual episode titles only...")
            else:
                if display:
                    display.add_ocr_diagnostic("TMDB API key not provided, skipping TMDB lookup")
        except Exception as e:
            tmdb_status['error_message'] = f"Unexpected error: {e}"
            if display:
                display.add_ocr_diagnostic(f"Error fetching from TMDB: {e}")
                display.add_ocr_diagnostic("Continuing with manual episode titles only...")
    
    # Append manual titles
    if manual_titles:
        episode_titles.extend(manual_titles)
        if display:
            display.add_ocr_diagnostic(f"Added {len(manual_titles)} manual episode titles")
    
    if not episode_titles:
        # Build detailed error message based on what we know
        error_parts = [
            "\n",
            "ERROR: No episode titles found.\n",
            "\n",
            f"Show: {show_config.get('show_name', 'N/A')}\n",
            f"Season: {show_config.get('season', 'N/A')}\n",
            "\n"
        ]
        
        # Add TMDB-specific diagnostics
        if tmdb_status['api_key_provided']:
            if tmdb_status['api_key_valid'] is False:
                error_parts.append("TMDB API Issue: Invalid API key detected (401 Unauthorized).\n")
                error_parts.append("  → Check your API key: --tmdb-key YOUR_KEY\n")
                error_parts.append("  → Get a key from: https://www.themoviedb.org/settings/api\n")
            elif tmdb_status['api_key_valid'] is None:
                # API key validity unknown (could be network issue, etc.)
                if tmdb_status['error_message']:
                    error_parts.append(f"TMDB API Issue: {tmdb_status['error_message']}\n")
                else:
                    error_parts.append("TMDB API Issue: Unable to verify API key (may be network issue).\n")
            elif tmdb_status['show_found'] is False:
                error_parts.append("TMDB Issue: Show not found.\n")
                error_parts.append(f"  → '{show_config.get('show_name')}' was not found in TMDB\n")
                error_parts.append("  → Try a different show name or use --episodes-file\n")
            elif tmdb_status['season_found'] is False:
                error_parts.append("TMDB Issue: Season not found.\n")
                error_parts.append(f"  → Season {show_config.get('season')} not found for this show\n")
                error_parts.append("  → Verify the season number or use --episodes-file\n")
            elif tmdb_status['error_message']:
                error_parts.append(f"TMDB Issue: {tmdb_status['error_message']}\n")
            else:
                error_parts.append("TMDB Issue: No episodes found.\n")
        else:
            error_parts.append("TMDB API key not provided.\n")
        
        error_parts.extend([
            "\n",
            "Solutions:\n",
            "  1. Provide a TMDB API key:\n",
            "     --tmdb-key YOUR_API_KEY\n",
            "     (or set TMDB_API_KEY environment variable)\n",
            "     Get a key from: https://www.themoviedb.org/settings/api\n",
            "\n",
            "  2. Provide an episodes file:\n",
            "     --episodes-file path/to/episodes.json\n",
            "\n"
        ])
        
        error_msg = ''.join(error_parts)
        
        if display:
            # Show error prominently in the display
            display.show_error(error_msg)
            # Add to diagnostics as well
            display.add_ocr_diagnostic("ERROR: No episode titles found")
            # Give display time to show the error (5 seconds so user can read it)
            time.sleep(5.0)
        else:
            # If no display, use rich console for error output
            console = Console(stderr=True)
            console.print(f"[red]{error_msg}[/red]")
        
        sys.exit(1)
    
    return episode_titles


def get_episode_info(config: Dict, display: Optional[OCRProgressDisplay] = None) -> Dict[str, Tuple[int, int]]:
    """
    Get episode information (season, episode number) from TMDB and/or config.
    Returns a dict mapping episode_title -> (season, episode_number).
    """
    show_config = config.get("show", {})
    episode_info = {}  # episode_title -> (season, episode_number)
    
    show_name = show_config.get("show_name")
    season = show_config.get("season")
    tmdb_api_key = show_config.get("tmdb_api_key")
    
    # Support new format: episodes array with explicit episode numbers
    episodes = show_config.get("episodes", [])
    # Support old format: flat episode_titles list (backward compatibility)
    episode_titles_old = show_config.get("episode_titles", [])
    
    # Try to fetch from TMDB if show_name and season are provided
    if show_name and season:
        try:
            # Get API key from config or environment
            api_key = tmdb_api_key
            if not api_key:
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
    
    # Process manual episodes from config
    if episodes:
        # New format: episodes array with explicit episode numbers
        for ep_entry in episodes:
            if isinstance(ep_entry, dict):
                ep_num = ep_entry.get("episode")
                ep_title = ep_entry.get("title")
                if ep_num is not None and ep_title:
                    # Use explicit episode number from config
                    # TMDB takes precedence if title already exists
                    if ep_title not in episode_info:
                        episode_info[ep_title] = (season, ep_num)
            elif isinstance(ep_entry, str):
                # Fallback: treat as title with sequential numbering
                # Only if not already in episode_info (TMDB takes precedence)
                if ep_entry not in episode_info:
                    # Use index in episodes array + 1 as episode number
                    ep_num = episodes.index(ep_entry) + 1
                    episode_info[ep_entry] = (season, ep_num)
    elif episode_titles_old:
        # Old format: flat list - assign sequential episode numbers starting from 1
        # Only assign if not already in episode_info (TMDB takes precedence)
        for idx, title in enumerate(episode_titles_old, start=1):
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
    console = Console()
    matched_count = sum(1 for _, new_name, _ in rename_preview if new_name)
    total_count = len(rename_preview)
    
    # Stop the main display cleanly (this will exit the main app)
    display.stop()
    
    # Wait a moment for the display thread to fully stop
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
    console = Console()
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
                console.print(f"[yellow]⚠[/yellow] Skipping {original}: target already exists")
                continue
            
            try:
                original_path.rename(new_path)
                renamed_count += 1
            except Exception as e:
                console.print(f"[red]Error renaming {original}:[/red] {e}")
    
    console.print(f"\n[green]✓[/green] Renamed {renamed_count} files")


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
    
    # Use standard formatter by default, only use rich formatter for help
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
    
    # Override print_help to use rich formatting only when help is requested
    def print_help(file=None):
        # Create a rich formatter just for this help output
        rich_formatter = RichHelpFormatter(parser.prog)
        # Temporarily replace the formatter
        original_formatter = parser._get_formatter()
        parser._formatter = rich_formatter
        try:
            # Generate help text with rich markup
            help_text = parser.format_help()
            # Print using rich console (which understands the markup)
            console = Console()
            console.print(help_text)
        finally:
            # Restore original formatter
            parser._formatter = original_formatter
    
    parser.print_help = print_help
    
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
        "--max-dimension",
        type=int,
        default=800,
        help="Maximum frame dimension in pixels (default: 800, use 0 for full resolution)"
    )
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
        default=0.6,
        help="Episode matching threshold 0.0-1.0 (default: 0.6, lower = more lenient)"
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
    
    # Load episodes file if provided (only for custom episodes)
    episodes = []
    if args.episodes_file:
        episodes_path = Path(args.episodes_file).expanduser()
        if not episodes_path.exists():
            console = Console(stderr=True)
            console.print(f"[red]Error:[/red] Episodes file not found: {episodes_path}")
            sys.exit(1)
        episodes_config = load_config(episodes_path)
        episodes = episodes_config.get("episodes", [])
    
    # Build show config from CLI arguments
    show_config = {
        "show_name": args.show,
        "season": args.season,
        "input_directory": args.input,
    }
    if args.tmdb_key:
        show_config["tmdb_api_key"] = args.tmdb_key
    if episodes:
        show_config["episodes"] = episodes
    
    # Build OCR config from CLI arguments (CLI is the only source)
    ocr_config = {
        "max_dimension": args.max_dimension if args.max_dimension > 0 else None,
        "duration": args.duration if args.duration > 0 else None,
        "frame_interval": args.frame_interval,
        "match_threshold": args.match_threshold,
        "hwaccel": args.hwaccel,
    }
    
    # Build config dict for use by helper functions
    config = {"show": show_config}
    
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
        # Get episode titles
        episode_titles = get_episode_titles(config, display)
        if display:
            display.add_ocr_diagnostic(f"Total episode titles: {len(episode_titles)}")
        
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
        early_exit = False
        processor = None  # Initialize to None so it's in scope for exception handler
        try:
            # Get episode info early so we can use it in status callback
            episode_info = get_episode_info(config, display)
            
            matches = process_video_files(input_directory, episode_titles, ocr_config, display, args.dry_run, processor_ref=None, episode_info=episode_info, show_name=show_name)
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


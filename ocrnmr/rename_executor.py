"""Rename preview and execution logic."""

import time
import uuid
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ocrnmr.display import OCRProgressDisplay
from ocrnmr.filename import sanitize_filename


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


def display_rename_preview(
    rename_preview: List[Tuple[str, Optional[str], Optional[str]]], 
    display: OCRProgressDisplay, 
    show_name: str, 
    season: int
):
    """Display rename preview using rich console formatting."""
    console = Console()
    matched_count = sum(1 for _, new_name, _ in rename_preview if new_name)
    total_count = len(rename_preview)
    
    # Stop the main display cleanly
    display.stop()
    
    # Wait a moment for the display thread to fully stop
    time.sleep(0.2)
    
    # Clear screen and print preview
    console.print()
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



def _resolve_unique_path(target_path: Path, existing_paths: Set[Path]) -> Path:
    """
    Resolve a target path to be unique among existing paths by appending a counter.
    
    Args:
        target_path: The desired target path
        existing_paths: Set of paths that already exist or are reserved
    
    Returns:
        A unique path
    """
    if target_path not in existing_paths and not target_path.exists():
        return target_path
        
    # If path exists, append counter
    base_name = target_path.stem
    suffix = target_path.suffix
    parent = target_path.parent
    
    counter = 1
    while True:
        new_name = f"{base_name} ({counter}){suffix}"
        new_path = parent / new_name
        if new_path not in existing_paths and not new_path.exists():
            return new_path
        counter += 1


def execute_renames(
    rename_preview: List[Tuple[str, Optional[str], Optional[str]]], 
    input_directory: Path
):
    """
    Execute file renames atomically with temporary intermediate files.
    
    Strategy:
    1. Rename all targets to temporary unique names (to handle case collisions and swaps).
    2. Rename all temporary files to their final destination.
    3. Handle duplicates by appending counters.
    """
    console = Console()
    renamed_count = 0
    
    # Filter only files that need renaming
    to_rename = []
    for original, new_name, _ in rename_preview:
        if new_name and original != new_name:
            to_rename.append((original, new_name))
            
    if not to_rename:
        console.print("[yellow]No files to rename.[/yellow]")
        return

    console.print(f"[bold]Executing {len(to_rename)} renames...[/bold]")
    
    # Step 1: Rename to temporary files
    temp_map: List[Tuple[Path, str]] = []  # List of (temp_path, final_name)
    
    try:
        with console.status("[bold green]Phase 1: Renaming to temporary files..."):
            for original, new_name in to_rename:
                original_path = input_directory / original
                
                if not original_path.exists():
                    console.print(f"[red]Error:[/red] File not found: {original}")
                    continue
                
                # Create a unique temp name in the same directory
                temp_name = f".tmp-{uuid.uuid4()}{original_path.suffix}"
                temp_path = input_directory / temp_name
                
                try:
                    original_path.rename(temp_path)
                    temp_map.append((temp_path, new_name))
                except Exception as e:
                    console.print(f"[red]Error creating temp file for {original}:[/red] {e}")
        
        # Step 2: Rename from temp to final
        # Track reserved paths to avoid collisions within this batch
        reserved_paths: Set[Path] = set()
        
        with console.status("[bold green]Phase 2: Renaming to final filenames..."):
            for temp_path, new_name in temp_map:
                target_path = input_directory / new_name
                
                # Resolve conflicts
                final_path = _resolve_unique_path(target_path, reserved_paths)
                reserved_paths.add(final_path)
                
                try:
                    temp_path.rename(final_path)
                    renamed_count += 1
                    
                    # Log if we had to change the name due to conflict
                    if final_path.name != new_name:
                        console.print(f"[yellow]Conflict resolved:[/yellow] {new_name} -> {final_path.name}")
                        
                except Exception as e:
                    console.print(f"[red]Error finalizing {new_name}:[/red] {e}")
                    # Try to revert to original name if possible? 
                    # It's hard to know original name here easily without storing it, 
                    # but leaving it as .tmp is safer than overwriting something else.
                    console.print(f"[yellow]File left at temporary path:[/yellow] {temp_path}")

    except Exception as e:
        console.print(f"[bold red]Critical Error during rename process:[/bold red] {e}")
    
    console.print(f"\n[green]✓[/green] Renamed {renamed_count} files")



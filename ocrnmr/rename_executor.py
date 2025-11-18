"""Rename preview and execution logic."""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def execute_renames(
    rename_preview: List[Tuple[str, Optional[str], Optional[str]]], 
    input_directory: Path
):
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


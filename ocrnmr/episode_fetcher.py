"""Episode fetching from TMDB and episode files."""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    requests = None

from ocrnmr.tmdb_client import TMDBClient


def load_episodes_file(file_path: Path) -> List[Dict]:
    """Load episodes from JSON file."""
    if not file_path.exists():
        return []
    
    try:
        with open(file_path, 'r') as f:
            config = json.load(f)
        return config.get("episodes", [])
    except (json.JSONDecodeError, KeyError):
        return []


def get_tmdb_episodes(show_name: str, season: int, api_key: Optional[str], display=None) -> Tuple[List[str], Dict[str, Tuple[int, int]]]:
    """
    Fetch episode titles and info from TMDB.
    
    Returns:
        Tuple of (episode_titles_list, episode_info_dict) where episode_info maps title -> (season, episode_num)
    """
    if not api_key:
        api_key = os.getenv('TMDB_API_KEY')
    
    if not api_key:
        if display:
            display.add_log("TMDB API key not provided, skipping TMDB lookup")
        return [], {}
    
    try:
        if display:
            display.add_log(f"Fetching episodes from TMDB for {show_name} Season {season}...")
        
        client = TMDBClient(api_key=api_key)
        episode_info_list = client.get_episode_info(show_name, season)
        
        if not episode_info_list:
            if display:
                display.add_log("No episodes found in TMDB")
            return [], {}
        
        # Build titles list and info dict
        titles = []
        info_dict = {}
        for ep_num, ep_title in episode_info_list:
            titles.append(ep_title)
            info_dict[ep_title] = (season, ep_num)
        
        if display:
            display.add_log(f"Found {len(titles)} episodes from TMDB")
        
        return titles, info_dict
        
    except requests.exceptions.HTTPError as e:
        if hasattr(e, 'response') and e.response.status_code == 401:
            error_msg = "Invalid TMDB API key (401 Unauthorized)"
            if display:
                display.add_log(f"Error: {error_msg}")
            raise ValueError(error_msg) from e
        else:
            error_msg = f"TMDB API error: {e}"
            if display:
                display.add_log(error_msg)
            raise ValueError(error_msg) from e
    except Exception as e:
        error_msg = f"Error accessing TMDB: {e}"
        if display:
            display.add_log(error_msg)
        raise ValueError(error_msg) from e


def get_manual_episodes(episodes: List[Dict], season: int) -> Tuple[List[str], Dict[str, Tuple[int, int]]]:
    """
    Extract episode titles and info from manual episodes list.
    
    Args:
        episodes: List of episode dicts with 'episode' and 'title' keys
        season: Season number to use for episode info
    
    Returns:
        Tuple of (episode_titles_list, episode_info_dict)
    """
    titles = []
    info_dict = {}
    
    for ep_entry in episodes:
        if isinstance(ep_entry, dict) and "title" in ep_entry:
            ep_title = ep_entry["title"]
            ep_num = ep_entry.get("episode")
            titles.append(ep_title)
            if ep_num is not None:
                info_dict[ep_title] = (season, ep_num)
    
    return titles, info_dict


def fetch_episodes(
    show_name: str,
    season: int,
    tmdb_api_key: Optional[str] = None,
    episodes_file: Optional[Path] = None,
    display=None
) -> Tuple[List[str], Dict[str, Tuple[int, int]]]:
    """
    Fetch episode titles and info from TMDB and/or episodes file.
    
    Args:
        show_name: TV show name
        season: Season number
        tmdb_api_key: Optional TMDB API key
        episodes_file: Optional path to episodes JSON file
        display: Optional display object for diagnostics
    
    Returns:
        Tuple of (episode_titles_list, episode_info_dict) where episode_info maps title -> (season, episode_num)
    
    Raises:
        ValueError: If no episodes found and TMDB lookup failed
        SystemExit: If episodes file is invalid
    """
    all_titles = []
    all_info = {}
    
    # Try TMDB first
    if show_name and season:
        try:
            tmdb_titles, tmdb_info = get_tmdb_episodes(show_name, season, tmdb_api_key, display)
            all_titles.extend(tmdb_titles)
            all_info.update(tmdb_info)
        except ValueError:
            # TMDB failed, continue with manual episodes if available
            pass
    
    # Add manual episodes from file
    if episodes_file:
        episodes = load_episodes_file(episodes_file)
        if episodes:
            manual_titles, manual_info = get_manual_episodes(episodes, season)
            # Only add titles not already in all_titles (TMDB takes precedence)
            for title in manual_titles:
                if title not in all_titles:
                    all_titles.append(title)
            # Update info dict (TMDB takes precedence)
            for title, info in manual_info.items():
                if title not in all_info:
                    all_info[title] = info
            
            if display:
                display.add_log(f"Added {len(manual_titles)} manual episode titles")
    
    # Validate we have episodes
    if not all_titles:
        error_msg = (
            f"\nERROR: No episode titles found.\n\n"
            f"Show: {show_name}\n"
            f"Season: {season}\n\n"
            f"Solutions:\n"
            f"  1. Provide a TMDB API key:\n"
            f"     --tmdb-key YOUR_API_KEY\n"
            f"     (or set TMDB_API_KEY environment variable)\n"
            f"     Get a key from: https://www.themoviedb.org/settings/api\n\n"
            f"  2. Provide an episodes file:\n"
            f"     --episodes-file path/to/episodes.json\n\n"
        )
        
        if display:
            display.show_error(error_msg)
            display.add_log("ERROR: No episode titles found")
            import time
            time.sleep(5.0)
        else:
            from rich.console import Console
            console = Console(stderr=True)
            console.print(f"[red]{error_msg}[/red]")
        
        sys.exit(1)
    
    return all_titles, all_info


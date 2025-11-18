"""TMDB API client for fetching TV show and episode information."""

from typing import Optional, Dict, List, Tuple
import os
import requests


class TMDBClient:
    BASE_URL = "https://api.themoviedb.org/3"

    def __init__(self, api_key: Optional[str] = None):
        """Initialize TMDB client with API key from environment or parameter."""
        # Try to get API key from environment if not provided
        self.api_key = api_key or os.getenv('TMDB_API_KEY')
        if not self.api_key:
            raise ValueError("TMDB API key must be provided or set in TMDB_API_KEY environment variable")
        
        self.session = requests.Session()
        self.session.headers = {
            'accept': 'application/json',
        }

    def search_tv_show(self, show_name: str) -> List[Dict]:
        """
        Search for a TV show and return list of matches.
        Returns list of dicts with show information.
        """
        url = f"{self.BASE_URL}/search/tv"
        params = {
            'api_key': self.api_key,
            'query': show_name
        }
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        results = response.json().get('results', [])
        return [
            {
                'id': show['id'],
                'name': show['name'],
                'first_air_date': show.get('first_air_date', ''),
                'overview': show.get('overview', '')
            }
            for show in results
        ]

    def get_season_episode_count(self, show_id: int, season: int) -> Optional[int]:
        """Get the total number of episodes in a season."""
        try:
            url = f"{self.BASE_URL}/tv/{show_id}/season/{season}"
            response = self.session.get(url, params={'api_key': self.api_key})
            response.raise_for_status()
            
            season_details = response.json()
            return len(season_details.get('episodes', []))
        except Exception as e:
            # Error handled silently - errors are shown via rich display at higher levels
            return None
    
    def get_season(self, show_id: int, season: int) -> Optional[Dict]:
        """Fetch full season details (single request) including all episodes."""
        try:
            url = f"{self.BASE_URL}/tv/{show_id}/season/{season}"
            response = self.session.get(url, params={'api_key': self.api_key})
            response.raise_for_status()
            return response.json()
        except Exception as e:
            # Error handled silently - errors are shown via rich display at higher levels
            return None
    
    def get_show_details(self, show_id: int) -> Optional[Dict]:
        """Fetch TV show details including all seasons."""
        try:
            url = f"{self.BASE_URL}/tv/{show_id}"
            response = self.session.get(url, params={'api_key': self.api_key})
            response.raise_for_status()
            return response.json()
        except Exception as e:
            # Error handled silently - errors are shown via rich display at higher levels
            return None
    
    def get_all_seasons(self, show_id: int) -> List[Dict]:
        """Get list of all seasons for a TV show."""
        try:
            show_details = self.get_show_details(show_id)
            if not show_details:
                return []
            
            seasons = show_details.get('seasons', [])
            # Filter out seasons with 0 episodes (specials, etc.) if desired
            # For now, include all seasons
            return seasons
        except Exception as e:
            # Error handled silently - errors are shown via rich display at higher levels
            return []

    def get_episode_info(self, show_name: str, season: int) -> List[Tuple[int, str]]:
        """
        Get all episode information for a show and season.
        Returns list of tuples (episode_number, episode_title) in order.
        """
        shows = self.search_tv_show(show_name)
        if not shows:
            return []
        
        # Use the first match
        show_id = shows[0]['id']
        season_data = self.get_season(show_id, season)
        
        if not season_data:
            return []
        
        episodes = season_data.get('episodes', [])
        # Sort by episode number to ensure correct order
        episodes.sort(key=lambda x: x.get('episode_number', 0))
        
        return [(ep.get('episode_number', 0), ep.get('name', '')) 
                for ep in episodes if ep.get('name')]


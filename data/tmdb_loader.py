import os
from core.logger import Logger
from dotenv import load_dotenv
from tmdbv3api import TMDb, Movie, TV
from typing import Dict, Optional

load_dotenv()

class TMDbLoader:
    """Loader class for TMDb data, fetching media details by ID."""
    def __init__(self):
        self.tmdb = TMDb()
        self.tmdb.api_key = os.getenv("TMDB_API_KEY")
        self.tmdb.language = "en"
        self.tmdb.debug = False
        self.logger = Logger().get_logger("TMDbLoader")

        self.movie_api = Movie()
        self.tv_api = TV()

    def fetch_tmdb_data(self, media_id: int, max_actors=5, delay=0.1, verbose: bool = False) -> Optional[Dict]:
        """Fetch TMDb data by media ID"""
        try:
            # time.sleep(delay)  # Unnecessary as it is handled by the library
            details = self.movie_api.details(media_id)
            if details:
                media_type = 'movie'
                credits = self.movie_api.credits(media_id)
            else:
                details = self.tv_api.details(media_id)
                media_type = 'tv'
                credits = self.tv_api.credits(media_id)

            cast = credits['cast']
            description = details.overview
            actors = [p['name'] for p in cast][:max_actors]
            director = None
            poster_path = details.poster_path

            for crew in credits.get("crew", []):
                if crew["job"] == "Director":
                    director = crew["name"]
                    break

            if verbose:
                self.logger.info(f"Fetched TMDb data for media ID {media_id}: {details.title} ({media_type})")

            return {
                "description": description,
                "actors": actors,
                "director": director,
                "media_type": media_type,
                "poster_path": poster_path
            }

        except Exception as e:
            if verbose:
                self.logger.error(f"Error fetching TMDb data for media ID {media_id}: {e}")
            return None


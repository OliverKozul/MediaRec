import csv
import sqlite3

from typing import List, Optional
from core.logger import Logger

logger = Logger().get_logger("Media")

class Media:
    def __init__(
        self,
        media_id: int,
        title: str,
        genres: List[str],
        media_type: Optional[str] = None,  # 'movie' or 'tv', even though only movies are currently supported
        director: Optional[str] = None,
        actors: Optional[List[str]] = None,
        description: Optional[str] = None
    ):
        self.media_id = media_id
        self.title = title
        self.genres = genres
        self.media_type = media_type
        self.director = director
        self.actors = actors
        self.description = description

    def __repr__(self):
        return f"<Media id={self.media_id} title={self.title} type={self.media_type} director={self.director}>"

    @staticmethod
    def create_media(media_id: int, title: str, genres: List[str], media_type: Optional[str] = None, director: Optional[str] = None, actors: Optional[List[str]] = None, description: Optional[str] = None):
        conn = sqlite3.connect("media_recommender.db")
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO media (media_id, title, media_type, genres, description, director, actors) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (media_id, title, media_type, ",".join(genres), description, director, ",".join(actors) if actors else None)
        )
        conn.commit()
        conn.close()

    @staticmethod
    def populate_media_db(csv_path: str = "data/processed/large/complete_media_with_imgs.csv", db_path: str = "media_recommender.db") -> None:
        """Populate the media table in SQLite database with data from a CSV file."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        with open(csv_path, "r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            rows = [
            (
                int(row["media_id"]),
                row["title"],
                row["media_type"],
                row["genres"],
                row["description"],
                row["director"],
                row["actors"],
                f"https://image.tmdb.org/t/p/w342{row['poster_path']}"
            )
            for idx, row in enumerate(reader)
            # if idx < 15000
            ]

        cursor.executemany(
            """
            INSERT OR IGNORE INTO media (media_id, title, media_type, genres, description, director, actors, poster_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows
        )

        conn.commit()
        conn.close()
        logger.info(f"Inserted {len(rows)} rows into the media table.")

    @staticmethod
    def get_media(media_id: int) -> Optional[tuple]:
        conn = sqlite3.connect("media_recommender.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM media WHERE media_id = ?", (media_id,))
        media_data = cursor.fetchone()
        conn.close()
        return media_data
    
    @staticmethod
    def get_all_media() -> List[tuple]:
        conn = sqlite3.connect("media_recommender.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM media")
        media_data = cursor.fetchall()
        conn.close()
        return media_data

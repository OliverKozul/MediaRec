import sqlite3
from typing import Dict, List, Optional
from core.logger import Logger
from werkzeug.security import generate_password_hash

logger = Logger().get_logger("User")

class User:
    def __init__(self, user_id: int):
        self.user_id = user_id

    def __repr__(self):
        return f"<User id={self.user_id}>"

    @staticmethod
    def create_user(username: str, password: str) -> Dict[str, int]:
        conn = sqlite3.connect("media_recommender.db")
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
        existing_user = cursor.fetchone()
        if existing_user:
            logger.warning(f"User creation failed: Username '{username}' already exists.")
            conn.close()
            return {"success": False, "user_id": User.get_user(username=username)['user_id']}

        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, generate_password_hash(password))
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        logger.info(f"User created: Username '{username}', ID {user_id}.")
        return {"success": True, "user_id": user_id}

    @staticmethod
    def get_user(user_id: Optional[int] = None, username: Optional[str] = None) -> Optional[Dict]:
        if not user_id and not username:
            raise ValueError("Either user_id or username must be provided.")

        conn = sqlite3.connect("media_recommender.db")
        cursor = conn.cursor()

        if user_id:
            cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        elif username:
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))

        user_data = cursor.fetchone()
        conn.close()
        return {k: v for k, v in zip(["user_id", "username", "password"], user_data)} if user_data else None

    @staticmethod
    def get_ratings(user_id: int) -> Dict[int, int]:
        conn = sqlite3.connect("media_recommender.db")
        cursor = conn.cursor()
        cursor.execute("SELECT media_id, rating FROM ratings WHERE user_id = ?", (user_id,))
        ratings = cursor.fetchall()
        conn.close()
        return {media_id: rating for media_id, rating in ratings}
    
    @staticmethod
    def get_all_media_for_user(user_id: int, rating: Optional[int] = None) -> List[Dict]:
        conn = sqlite3.connect("media_recommender.db")
        cursor = conn.cursor()
        if rating is not None:
            cursor.execute("""
                SELECT m.media_id, m.title, m.media_type, m.genres, m.description, m.director, m.actors, r.rating
                FROM media AS m
                JOIN ratings AS r ON m.media_id = r.media_id
                WHERE r.user_id = ? AND r.rating >= ?
            """, (user_id, rating))
        else:
            cursor.execute("""
                SELECT m.media_id, m.title, m.media_type, m.genres, m.description, m.director, m.actors, r.rating
                FROM media AS m
                JOIN ratings AS r ON m.media_id = r.media_id
                WHERE r.user_id = ?
            """, (user_id,))
        media_data = cursor.fetchall()
        conn.close()
        return [
            {
                "media_id": media_id,
                "title": title,
                "media_type": media_type,
                "genres": genres,
                "description": description,
                "director": director,
                "actors": actors,
                "rating": rating
            }
            for media_id, title, media_type, genres, description, director, actors, rating in media_data
        ]

    @staticmethod
    def rate(user_id: int, media_id: int, rating: int) -> None:
        if 1 <= rating <= 5:
            conn = sqlite3.connect("media_recommender.db")
            cursor = conn.cursor()
            cursor.execute(
                "REPLACE INTO ratings (user_id, media_id, rating) VALUES (?, ?, ?)",
                (user_id, media_id, rating)
            )
            conn.commit()
            conn.close()
            logger.info(f"Rating saved: User ID {user_id}, Media ID {media_id}, Rating {rating}.")
        else:
            raise ValueError("Rating must be between 1 and 5.")

    @staticmethod
    def delete_rating(user_id: int, media_id: int) -> None:
        conn = sqlite3.connect("media_recommender.db")
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM ratings WHERE user_id = ? AND media_id = ?",
            (user_id, media_id)
        )
        conn.commit()
        conn.close()
        logger.info(f"Rating deleted: User ID {user_id}, Media ID {media_id}.")

    @staticmethod
    def delete_all_ratings(user_id: int) -> None:
        conn = sqlite3.connect("media_recommender.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM ratings WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()
        logger.info(f"All ratings deleted for User ID {user_id}.")

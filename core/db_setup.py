import sqlite3
from models.media import Media
from models.user import User

def initialize_db() -> None:
    conn = sqlite3.connect("media_recommender.db")
    cursor = conn.cursor()

    # Create users table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            username TEXT UNIQUE,
            password TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.commit()
    # Create default admin user if it does not already exist
    User.create_user(username="admin", password="admin123")

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='media'")
    media_table_exists = cursor.fetchone()

    # Create media table if it doesn't exist
    if not media_table_exists:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS media (
                media_id INTEGER PRIMARY KEY,
                title TEXT,
                media_type TEXT,
                genres TEXT,
                description TEXT,
                director TEXT,
                actors TEXT,
                poster_path TEXT
            )
            """
        )
        conn.commit()

        # Populate media table
        Media.populate_media_db()

    # Create ratings table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ratings (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            media_id INTEGER,
            rating INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            FOREIGN KEY (media_id) REFERENCES media(media_id),
            UNIQUE(user_id, media_id)
        )
        """
    )

    conn.commit()
    conn.close()

if __name__ == "__main__":
    initialize_db()

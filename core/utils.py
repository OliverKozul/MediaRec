import pandas as pd
from typing import Dict
from recommender.engine import RecommendationEngine
from models.user import User
from models.media import Media
from core.logger import Logger

logger = Logger().get_logger("Utils")

def summarize_user_preferences(user_id: int) -> Dict[str, Dict]:
    """Summarize user preferences based on their ratings. Used in user profile page."""
    ratings = User.get_ratings(user_id)

    if not ratings:
        return {
            "Top Genres": {},
            "Top Directors": {},
            "Top Actors": {},
        }

    rated = []
    for media_id, rating in ratings.items():
        media = Media.get_media(media_id)
        if media:
            rated.append({
                "rating": rating,
                "genres": media[3].split("|"),
                "director": media[5],
                "actors": media[6].split(",")
            })

    df = pd.DataFrame(rated)

    genre_scores = df.explode("genres").groupby("genres")["rating"].mean().sort_values(ascending=False)
    director_scores = df.groupby("director")["rating"].mean().sort_values(ascending=False)
    actor_scores = df.explode("actors").groupby("actors")["rating"].mean().sort_values(ascending=False)

    return {
        "Top Genres": genre_scores.head(5).to_dict(),
        "Top Directors": director_scores.head(5).to_dict(),
        "Top Actors": actor_scores.head(5).to_dict(),
    }

def perform_test() -> None:
    """Run a basic test of the recommendation engine."""
    recommendation_engine = RecommendationEngine()
    recommendation_engine.load_model()
    recommendation_engine.load_vectorizers()

    user_id = User.create_user(username="Oli", password="password123")['user_id']
    user = User(user_id=user_id)
    
    user.rate(media_id=161, rating=4)
    user.rate(media_id=188, rating=5)
    user.rate(media_id=798, rating=4)
    
    recs = recommendation_engine.recommend(user_id=user_id, top_n=5)

    logger.info("\nRecommendations:")
    for rec in recs:
        logger.info(f"{rec['title']} — Predicted rating: {rec['predicted_rating']:.2f}")
        explanation = recommendation_engine.explain_single_prediction(user_id, pd.Series(rec))
        logger.info(f"Explanation: {explanation}")


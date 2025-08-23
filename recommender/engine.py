import pandas as pd
import numpy as np
import joblib
import os
import shap

from typing import Union
from scipy.sparse import hstack
from models.user import User
from models.media import Media
from core.logger import Logger
from data.data_processor import (
    process_media_dataset,
    process_genres,
    process_actors,
    process_director
)

MODEL_DIR = "recommender/model"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")
MLB_PATH = os.path.join(MODEL_DIR, "mlb_genres.pkl")
ACTORS_VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_actors.pkl")
DIRECTORS_VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_directors.pkl")
DESC_VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_desc.pkl")

class RecommendationEngine:
    """Engine for generating media recommendations based on user ratings and media features."""
    def __init__(self):
        self.model = None
        self.vectorizers = {}
        self.logger = Logger().get_logger("RecommendationEngine")
        self._media_df = None
        self._media_vectors = None
        self._media_id_to_idx = None

    def load_model(self, model_path: str = MODEL_PATH):
        try:
            self.model = joblib.load(model_path)
            self.logger.info("Model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load model from {model_path}: {e}")
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    def cache_vectorized_media(self):
        """Vectorize all media and cache the results for fast recommendation. Persist cache to disk."""
        cache_dir = MODEL_DIR
        df_path = os.path.join(cache_dir, "media_df.pkl")
        vec_path = os.path.join(cache_dir, "media_vectors.npy")

        if os.path.exists(df_path) and os.path.exists(vec_path):
            self.logger.info("Loading cached media vectors and DataFrame from disk...")
            media_df = pd.read_pickle(df_path)
            media_vectors = np.load(vec_path)
        else:
            self.logger.info("No cache found. Vectorizing media and saving cache...")
            all_media = Media.get_all_media()
            media_df = pd.DataFrame(all_media, columns=[
                "media_id", "title", "media_type", "genres", "description", "director", "actors", "poster_path"
            ])
            media_df['original_director'] = media_df['director']
            media_df = media_df.apply(process_media_dataset, axis=1)
            media_df['title'] = media_df['title'].astype(str)
            media_vectors = self.vectorize_media(media_df)
            media_df.to_pickle(df_path)
            np.save(vec_path, media_vectors)
            self.logger.info(f"Saved media_df to {df_path} and media_vectors to {vec_path}.")

        self._media_df = media_df
        self._media_vectors = media_vectors
        self._media_id_to_idx = {mid: idx for idx, mid in enumerate(media_df['media_id'])}
        self.logger.info(f"Cached vectorized media dataset with {len(media_df)} items.")

    def load_vectorizers(self):
        self.vectorizers = {
            "mlb": joblib.load(MLB_PATH),
            "actors": joblib.load(ACTORS_VECTORIZER_PATH),
            "directors": joblib.load(DIRECTORS_VECTORIZER_PATH),
            # "description": joblib.load(DESC_VECTORIZER_PATH), # Not used in current implementation
        }
        self.logger.info("Vectorizers loaded successfully.")

    def get_feature_names(self) -> list:
        """Get feature names from all vectorizers"""
        feature_names = (
            self.vectorizers["mlb"].classes_.tolist() + 
            self.vectorizers["actors"].get_feature_names_out().tolist() + 
            self.vectorizers["directors"].get_feature_names_out().tolist()
            # self.vectorizers["description"].get_feature_names_out().tolist()
        )

        return feature_names

    def vectorize_media(self, df: pd.DataFrame) -> np.ndarray:
        genres_split = df["genres"].str.split("|")
        X_genres = self.vectorizers["mlb"].transform(genres_split)
        X_actors = self.vectorizers["actors"].transform(df["actors"])
        X_directors = self.vectorizers["directors"].transform(df["director"])
        # X_desc = self.vectorizers["description"].transform(df["description"])

        return hstack([X_genres, X_actors, X_directors]).toarray()

    def compute_similarity_scores(self, media_df: pd.DataFrame, user_df: pd.DataFrame, actor_weight: float = 0.15, director_weight: float = 0.15, title_weight: float = 0.2) -> np.ndarray:
        """Vectorized similarity score for all media in media_df against user_df."""
        liked_actors = set()
        disliked_actors = set()
        liked_directors = set()
        disliked_directors = set()
        liked_titles = set()
        disliked_titles = set()
        for _, row in user_df.iterrows():
            actors = {a.strip().lower().replace(' ', '') for a in str(row['actors']).split(',') if a.strip()}
            director = str(row['director']).strip().lower()
            title = str(row['title']).lower()
            if row['rating'] >= 4:
                liked_actors.update(actors)
                liked_directors.add(director)
                liked_titles.add(title)
            elif row['rating'] <= 2:
                disliked_actors.update(actors)
                disliked_directors.add(director)
                disliked_titles.add(title)

        media_actors = media_df['actors'].apply(lambda x: {a.strip().lower().replace(' ', '') for a in str(x).split(',') if a.strip()})
        media_directors = media_df['director'].str.strip().str.lower()
        media_titles = media_df['title'].str.lower()

        actor_sim = media_actors.apply(lambda actors: sum([(1 if a in liked_actors else -1 if a in disliked_actors else 0) for a in actors]) if actors else 0).to_numpy()

        director_sim = media_directors.apply(lambda d: 1.0 if d in liked_directors else -1.0 if d in disliked_directors else 0.0).to_numpy()

        def title_score(media_title):
            score = 0.0
            count = 0
            for t in liked_titles:
                if len(t) > 5 and (t in media_title or media_title in t):
                    score += 1.0
                    count += 1
            for t in disliked_titles:
                if len(t) > 5 and (t in media_title or media_title in t):
                    score -= 1.0
                    count += 1
            return score / count if count > 0 else 0.0
        title_sim = media_titles.apply(title_score).to_numpy()

        sim_score = actor_weight * actor_sim + director_weight * director_sim + title_weight * title_sim
        return sim_score

    def build_user_profile(self, user_id: int) -> np.ndarray:
        liked_df = pd.DataFrame(User.get_all_media_for_user(user_id, rating=4), columns=[
            "media_id", "title", "media_type", "genres", "description", "director", "actors", "rating"
        ])
        liked_df = liked_df.apply(process_media_dataset, axis=1)
        if liked_df.empty:
            self.logger.warning("Liked media DataFrame is empty.")
            return None
        liked_vectors = self.vectorize_media(liked_df)
        return liked_vectors.mean(axis=0)

    def recommend(self, user_id: int, top_n: int = 10, temperature: float = 0.0) -> list:
        if self._media_df is None or self._media_vectors is None or self._media_id_to_idx is None:
            self.cache_vectorized_media()

        temperature = (temperature + 1) / 2
        user_ratings = User.get_ratings(user_id)
        rated_ids = set(user_ratings.keys())

        unrated_mask = ~self._media_df["media_id"].isin(rated_ids)
        unrated_df = self._media_df[unrated_mask].copy()
        if unrated_df.empty:
            self.logger.info("No unrated media to recommend.")
            return []

        unrated_indices = unrated_df.index.tolist()
        media_vectors = self._media_vectors[unrated_indices]

        user_profile = self.build_user_profile(user_id)
        if user_profile is None:
            self.logger.info("Insufficient data to build user profile.")
            return []

        user_df = pd.DataFrame(User.get_all_media_for_user(user_id), columns=[
            "media_id", "title", "media_type", "genres", "description", "director", "actors", "rating"
        ])
        user_df = user_df.apply(process_media_dataset, axis=1)

        user_vectors = np.repeat(user_profile.reshape(1, -1), len(media_vectors), axis=0)
        X_input = np.hstack([user_vectors, media_vectors])

        top_recs = self.calculate_recommendations(X_input, unrated_df, user_df, top_n, temperature)
        self.logger.info(f"Top {top_n} recommendations generated for user {user_id}.")
        return top_recs.to_dict(orient="records")

    def calculate_recommendations(self, X_input: np.ndarray, unrated_df: pd.DataFrame, user_df: pd.DataFrame, top_n: int, temperature: float) -> pd.DataFrame:
        predicted_ratings = self.model.predict(X_input)
        unrated_df["predicted_rating"] = np.clip(predicted_ratings, 1.0, 5.0)

        if not user_df.empty:
            similarity_scores = self.compute_similarity_scores(unrated_df, user_df)
        else:
            similarity_scores = np.zeros(len(unrated_df))
        unrated_df["similarity_score"] = similarity_scores

        unrated_df["final_score"] = unrated_df["predicted_rating"] + temperature * (unrated_df["similarity_score"] - 1)
        unrated_df["final_score"] = np.clip(unrated_df["final_score"], 1.0, 5.0)

        top_recs = unrated_df.sort_values("final_score", ascending=False).head(top_n)
        top_recs["director"] = top_recs["original_director"]

        top_recs = top_recs[["media_id", "title", "description", "actors", "genres", "director", "predicted_rating", "similarity_score", "final_score", "poster_path"]]
        top_recs = top_recs.sort_values("final_score", ascending=False).reset_index(drop=True)

        return top_recs

    def explain_single_prediction(self, user_id: int, media_input: Union[pd.Series, pd.DataFrame], max_features: int = 10, temperature: float = 0.0) -> dict:
        if isinstance(media_input, pd.Series):
            media_df = pd.DataFrame([media_input])
        else:
            media_df = media_input
        self.logger.info(f"Explaining prediction for: {media_df['title'].values[0]}")
        temperature = (temperature + 1) / 2
        media_df['original_director'] = media_df['director']
        media_df = media_df.apply(process_media_dataset, axis=1)
        media_df['title'] = media_df['title'].astype(str)

        user_profile = self.build_user_profile(user_id)
        if user_profile is None:
            self.logger.warning("Insufficient data to build user profile for explanation.")
            return {}

        user_df = pd.DataFrame(User.get_all_media_for_user(user_id), columns=[
            "media_id", "title", "media_type", "genres", "description", "director", "actors", "rating"
        ])
        user_df = user_df.apply(process_media_dataset, axis=1)

        media_vectors = self.vectorize_media(media_df)
        user_vectors = np.repeat(user_profile.reshape(1, -1), len(media_vectors), axis=0)
        X_media = np.hstack([user_vectors, media_vectors])

        recs_df = self.calculate_recommendations(X_media, media_df, user_df, top_n=1, temperature=temperature)
        rec = recs_df.iloc[0]
        final_score = rec['final_score']

        explainer = shap.TreeExplainer(self.model)
        self.logger.info("Calculating SHAP values...")
        shap_values = explainer.shap_values(X_media, check_additivity=False)[0]
        self.logger.info("SHAP values calculated.")

        genre_features   = [f"[GENRE] {g}" for g in self.vectorizers["mlb"].classes_]
        actor_features   = [f"[ACTOR] {t}" for t in self.vectorizers["actors"].get_feature_names_out()]
        director_features= [f"[DIRECTOR] {t}" for t in self.vectorizers["directors"].get_feature_names_out()]
        feature_names    = genre_features + actor_features + director_features + []

        top_indices = np.argsort(np.abs(shap_values))[::-1][:max_features]
        explanation = {feature_names[idx]: shap_values[idx] for idx in top_indices if idx < len(feature_names)}
        orig_actors = media_df['actors'].values[0] if 'actors' in media_df else None
        orig_director = media_df['original_director'].values[0] if 'original_director' in media_df else None
        explanation_out = {}

        for k, v in explanation.items():
            if k.startswith('[ACTOR]') and orig_actors:
                processed = k.split('] ')[1]
                orig_match = next((a.strip() for a in str(orig_actors).split(',') if process_actors(a.strip()) == processed), processed)
                explanation_out[f'[ACTOR] {orig_match}'] = v
            elif k.startswith('[DIRECTOR]') and orig_director:
                processed = k.split('] ')[1]
                orig_match = orig_director if process_director(orig_director) == processed else processed
                explanation_out[f'[DIRECTOR] {orig_match}'] = v
            elif k.startswith('[GENRE]'):
                explanation_out[k] = v
            else:
                explanation_out[k] = v

        explanation_out = {k: v for k, v in explanation_out.items() if k.split('] ')[1].strip()[0].isupper()}
        for name, val in explanation_out.items():
            self.logger.info(f"  {name:<35} → SHAP = {val:+.3f}")

        return {
            "title": media_df["title"].values[0],
            "model_score": final_score,
            "explanation": explanation_out
        }

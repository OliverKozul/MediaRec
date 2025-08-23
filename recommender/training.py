import numpy as np
import joblib
import dask.dataframe as dd

from tqdm import tqdm
from core.logger import Logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from recommender.engine import (
    RecommendationEngine,
    MODEL_PATH,
    MLB_PATH,
    ACTORS_VECTORIZER_PATH,
    DIRECTORS_VECTORIZER_PATH,
    # DESC_VECTORIZER_PATH
)

from data.data_processor import (    
    process_genres,
    process_actors,
    process_director
)

class ModelTrainer:
    """Trainer class for building and evaluating the recommendation model."""
    def __init__(self, media_csv_path, ratings_csv_path):
        self.media_csv_path = media_csv_path
        self.ratings_csv_path = ratings_csv_path
        self.df = None
        self.engine = RecommendationEngine()
        self.all_media_features = None
        self.logger = Logger().get_logger("ModelTrainer")

    def load_data(self) -> None:
        self.logger.info("Loading datasets")
        media_df = dd.read_csv(self.media_csv_path).sample(frac=0.5, random_state=42).dropna(subset=["genres", "actors", "director", "description"])
        ratings_df = dd.read_csv(self.ratings_csv_path).sample(frac=0.1, random_state=42).dropna(subset=["user_id", "media_id", "rating"])
        self.df = dd.merge(ratings_df, media_df, on="media_id").dropna()
        self.df = self.df.drop(columns=["description", "poster_path", "timestamp"])
        self.logger.info("Datasets successfully loaded and merged")
        self.logger.info(f"Number of rows: {self.df.shape[0].compute()}")
        self.logger.info(f"Columns: {self.df.compute().columns.tolist()}")
        self.logger.info(f"DF Head:\n{self.df.compute().head()}")

    def fit_vectorizers(self) -> None:
        self.logger.info("Fitting vectorizers")
        self.df["genres"] = self.df["genres"].apply(process_genres, meta=("genres", "object"))
        genres_split = self.df["genres"].str.split("|")
        mlb = MultiLabelBinarizer()
        mlb.fit(genres_split.compute())
        joblib.dump(mlb, MLB_PATH)
        self.engine.vectorizers["mlb"] = mlb
        self.logger.info("MultiLabelBinarizer fitted and saved")

        self.df["actors"] = self.df["actors"].apply(process_actors, meta=("actors", "object"))
        tfidf_actors = TfidfVectorizer(
            strip_accents='unicode',
            max_features=500
        )
        tfidf_actors.fit(self.df["actors"].compute())
        joblib.dump(tfidf_actors, ACTORS_VECTORIZER_PATH)
        self.engine.vectorizers["actors"] = tfidf_actors
        self.logger.info("Actors TfidfVectorizer fitted and saved")

        self.df["director"] = self.df["director"].apply(process_director, meta=("director", "object"))
        tfidf_dir = TfidfVectorizer(
            max_features=250,
        )
        tfidf_dir.fit(self.df["director"].compute())
        joblib.dump(tfidf_dir, DIRECTORS_VECTORIZER_PATH)
        self.engine.vectorizers["directors"] = tfidf_dir
        self.logger.info("Directors TfidfVectorizer fitted and saved")

        # tfidf_desc = TfidfVectorizer(
        #     max_features=250,
        #     stop_words='english',
        #     strip_accents='unicode',
        #     ngram_range= (1, 3),
        #     tokenizer=space_tokenizer,
        #     preprocessor=identity_preprocessor,
        #     token_pattern=None,
        #     lowercase=True
        # )
        # tfidf_desc.fit(self.df["description"].compute())
        # joblib.dump(tfidf_desc, DESC_VECTORIZER_PATH)
        # self.engine.vectorizers["description"] = tfidf_desc
        # self.logger.info("Description TfidfVectorizer fitted and saved")

    def load_vectorizers(self) -> None:
        self.logger.info("Loading pre-fitted vectorizers")
        self.engine.vectorizers = {
            "mlb": joblib.load(MLB_PATH),
            "actors": joblib.load(ACTORS_VECTORIZER_PATH),
            "directors": joblib.load(DIRECTORS_VECTORIZER_PATH),
            # "description": joblib.load(DESC_VECTORIZER_PATH),
        }
        self.logger.info("Vectorizers loaded successfully")

    def vectorize_media(self) -> None:
        if not self.engine.vectorizers:
            self.logger.info("Loading vectorizers before vectorizing media dataset")
            self.load_vectorizers()
        self.logger.info("Vectorizing media dataset")
        self.all_media_features = self.engine.vectorize_media(self.df.compute())

    def train_model(self):
        self.logger.info("Preparing training data")
        self.df = self.df.compute()
        self.df["vector_index"] = range(len(self.df))
        grouped = self.df.groupby("user_id")

        X_combined = []
        y_combined = []

        for _, group in tqdm(grouped, desc="Building user training vectors"):
            liked = group[group["rating"] >= 4]
            if liked.empty:
                continue

            liked_vectors = self.all_media_features[liked["vector_index"]]
            user_profile = liked_vectors.mean(axis=0)

            for _, row in group.iterrows():
                media_vector = self.all_media_features[int(row["vector_index"])]
                combined = np.hstack([user_profile, media_vector])
                X_combined.append(combined)
                y_combined.append(row["rating"])

        X_combined = np.array(X_combined)
        y_combined = np.array(y_combined)

        X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, stratify=y_combined, test_size=0.2, random_state=42)

        model = XGBRegressor(
            n_estimators=250,
            max_depth=12,
            random_state=42,
            n_jobs=8,
            verbosity=1
        )
        # model = RandomForestRegressor(
        #     n_estimators=100,
        #     max_depth=12,
        #     min_samples_split=10,
        #     min_samples_leaf=5,
        #     random_state=42,
        #     n_jobs=8,
        #     verbose=1
        # )
        self.logger.info("Starting model training")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        self.logger.info(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.3f}")
        self.logger.info(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.3f}")
        self.logger.info(f"R^2 Score: {r2_score(y_test, y_pred):.3f}")

        joblib.dump(model, MODEL_PATH)
        self.logger.info(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    trainer = ModelTrainer(
        media_csv_path="data/processed/large/complete_media_with_imgs.csv",
        ratings_csv_path="data/processed/large/ratings.csv"
    )
    trainer.load_data()
    trainer.fit_vectorizers()
    trainer.vectorize_media()
    trainer.train_model()

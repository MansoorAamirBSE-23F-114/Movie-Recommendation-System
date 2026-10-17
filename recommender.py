"""Hybrid movie recommender: collaborative filtering + genre content similarity."""
import io
import os
import zipfile

import numpy as np
import pandas as pd
import requests
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = "data"


def download_data(data_dir=DATA_DIR):
    """Download MovieLens (small) once and keep movies.csv and ratings.csv."""
    files = ("movies.csv", "ratings.csv")
    if all(os.path.exists(os.path.join(data_dir, f)) for f in files):
        return
    os.makedirs(data_dir, exist_ok=True)
    resp = requests.get(URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        for name in files:
            with z.open(f"ml-latest-small/{name}") as src, \
                    open(os.path.join(data_dir, name), "wb") as dst:
                dst.write(src.read())


def build_model(data_dir=DATA_DIR):
    """Load data and build the content and collaborative matrices."""
    movies = pd.read_csv(os.path.join(data_dir, "movies.csv"))
    ratings = pd.read_csv(os.path.join(data_dir, "ratings.csv"))

    movies["genres"] = movies["genres"].replace("(no genres listed)", "")
    movies = movies.reset_index(drop=True)

    # Rating statistics per movie
    stats = ratings.groupby("movieId")["rating"].agg(avg_rating="mean", n_ratings="count")
    movies = movies.merge(stats, on="movieId", how="left")
    movies["avg_rating"] = movies["avg_rating"].fillna(0).round(2)
    movies["n_ratings"] = movies["n_ratings"].fillna(0).astype(int)

    # Content matrix: TF-IDF over genres (split on "|")
    tfidf = TfidfVectorizer(token_pattern=r"[^|]+", lowercase=True)
    content = tfidf.fit_transform(movies["genres"])

    # Collaborative matrix: movies x users, ratings centered per user
    ratings["centered"] = ratings["rating"] - ratings.groupby("userId")["rating"].transform("mean")
    movie_row = {mid: i for i, mid in enumerate(movies["movieId"])}
    rows = ratings["movieId"].map(movie_row).to_numpy()
    user_codes, _ = pd.factorize(ratings["userId"])
    collab = csr_matrix(
        (ratings["centered"].to_numpy(), (rows, user_codes)),
        shape=(len(movies), int(user_codes.max()) + 1),
    )
    return movies, content, collab


def recommend(movies, content, collab, liked_titles, n=10, cf_weight=0.7, min_ratings=10):
    """Return top-n movies similar to the liked titles as a DataFrame."""
    idx = movies.index[movies["title"].isin(liked_titles)].to_numpy()
    if len(idx) == 0:
        return movies.iloc[0:0][["title", "genres", "avg_rating", "n_ratings"]]

    cf = cosine_similarity(collab[idx], collab).mean(axis=0)
    cb = cosine_similarity(content[idx], content).mean(axis=0)
    score = cf_weight * cf + (1 - cf_weight) * cb

    score[idx] = -np.inf                                   # hide the liked movies
    score[movies["n_ratings"].to_numpy() < min_ratings] = -np.inf  # skip obscure ones

    top = np.argsort(-score)[:n]
    top = top[np.isfinite(score[top])]
    out = movies.iloc[top][["title", "genres", "avg_rating", "n_ratings"]].copy()
    out["match_score"] = score[top].round(3)
    return out.reset_index(drop=True)

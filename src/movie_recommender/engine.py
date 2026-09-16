"""Core recommendation logic for the MovieLens 100K application."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

GENRE_COLUMNS = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

ITEM_COLUMNS = [
    "movie_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url",
    *GENRE_COLUMNS,
]


def load_movielens(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the MovieLens 100K ratings and movie metadata files."""
    data_path = Path(data_dir)
    ratings_path = data_path / "u.data"
    movies_path = data_path / "u.item"

    missing = [str(path) for path in (ratings_path, movies_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing MovieLens file(s): {', '.join(missing)}")

    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
        dtype={"user_id": "int32", "movie_id": "int32", "rating": "float32", "timestamp": "int64"},
    )
    movies = pd.read_csv(
        movies_path,
        sep="|",
        encoding="latin-1",
        names=ITEM_COLUMNS,
        usecols=["movie_id", "title", "release_date", "imdb_url", *GENRE_COLUMNS],
    )

    if ratings.empty or movies.empty:
        raise ValueError("MovieLens files were loaded but contained no records.")
    if not ratings["rating"].between(1, 5).all():
        raise ValueError("Ratings must be on the MovieLens 1–5 scale.")

    title_year = pd.to_numeric(movies["title"].str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce")
    release_year = pd.to_datetime(movies["release_date"], errors="coerce").dt.year
    movies["year"] = title_year.fillna(release_year).astype("Int64")
    movies["genres"] = movies[GENRE_COLUMNS].apply(
        lambda row: ", ".join(column for column, value in row.items() if value == 1 and column != "unknown")
        or "Unknown",
        axis=1,
    )
    return ratings, movies


class MovieRecommender:
    """Adjusted-cosine item recommender with significance weighting and genre blending."""

    def __init__(self, shrinkage: float = 25.0) -> None:
        if shrinkage < 0:
            raise ValueError("shrinkage must be non-negative")
        self.shrinkage = float(shrinkage)
        self.catalog: pd.DataFrame | None = None
        self.collaborative_similarity: pd.DataFrame | None = None
        self.content_similarity: pd.DataFrame | None = None

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame) -> MovieRecommender:
        """Build collaborative and genre-similarity matrices."""
        required_ratings = {"user_id", "movie_id", "rating"}
        required_movies = {"movie_id", "title", *GENRE_COLUMNS}
        if not required_ratings.issubset(ratings.columns):
            raise ValueError(f"ratings must include {sorted(required_ratings)}")
        if not required_movies.issubset(movies.columns):
            raise ValueError("movies is missing required metadata or genre columns")

        valid_movie_ids = np.intersect1d(ratings["movie_id"].unique(), movies["movie_id"].unique())
        filtered_ratings = ratings[ratings["movie_id"].isin(valid_movie_ids)].copy()
        matrix = filtered_ratings.pivot_table(index="user_id", columns="movie_id", values="rating")

        # Adjust for users who consistently rate higher or lower than others.
        user_means = matrix.mean(axis=1)
        centered = matrix.sub(user_means, axis=0).fillna(0.0)
        raw_similarity = cosine_similarity(centered.T)

        # Discount high similarity scores supported by only a few shared raters.
        observed = matrix.notna().astype("int16")
        co_rating_counts = observed.T.dot(observed).to_numpy(dtype="float32")
        significance = co_rating_counts / (co_rating_counts + self.shrinkage)
        collaborative = raw_similarity * significance
        np.fill_diagonal(collaborative, 1.0)

        movie_order = matrix.columns.astype(int)
        self.collaborative_similarity = pd.DataFrame(
            collaborative,
            index=movie_order,
            columns=movie_order,
            dtype="float32",
        )

        movie_lookup = movies.drop_duplicates("movie_id").set_index("movie_id").loc[movie_order].copy()
        genre_vectors = movie_lookup[GENRE_COLUMNS].to_numpy(dtype="float32")
        content = cosine_similarity(genre_vectors)
        self.content_similarity = pd.DataFrame(content, index=movie_order, columns=movie_order, dtype="float32")

        stats = filtered_ratings.groupby("movie_id")["rating"].agg(rating_count="count", average_rating="mean")
        self.catalog = movie_lookup.join(stats).reset_index()
        self.catalog["rating_count"] = self.catalog["rating_count"].fillna(0).astype(int)
        self.catalog["average_rating"] = self.catalog["average_rating"].fillna(0.0)
        if "year" not in self.catalog:
            self.catalog["year"] = pd.Series(pd.NA, index=self.catalog.index, dtype="Int64")
        if "genres" not in self.catalog:
            self.catalog["genres"] = self.catalog[GENRE_COLUMNS].apply(
                lambda row: ", ".join(column for column, value in row.items() if value == 1 and column != "unknown")
                or "Unknown",
                axis=1,
            )
        return self

    @property
    def is_fitted(self) -> bool:
        return self.catalog is not None and self.collaborative_similarity is not None

    def recommend(
        self,
        seed_movie_ids: Iterable[int],
        n: int = 10,
        collaborative_weight: float = 0.85,
        min_rating_count: int = 10,
    ) -> pd.DataFrame:
        """Return ranked recommendations for one or more seed movies."""
        if not self.is_fitted or self.content_similarity is None or self.catalog is None:
            raise RuntimeError("Call fit before requesting recommendations.")
        if n < 1:
            raise ValueError("n must be at least 1")
        if not 0 <= collaborative_weight <= 1:
            raise ValueError("collaborative_weight must be between 0 and 1")

        seeds = list(dict.fromkeys(int(movie_id) for movie_id in seed_movie_ids))
        available = self.collaborative_similarity.columns
        unknown = sorted(set(seeds) - set(available))
        if not seeds:
            raise ValueError("Select at least one seed movie.")
        if unknown:
            raise KeyError(f"Unknown movie ID(s): {unknown}")

        collaborative_score = self.collaborative_similarity.loc[:, seeds].mean(axis=1)
        content_score = self.content_similarity.loc[:, seeds].mean(axis=1)
        score = collaborative_weight * collaborative_score + (1 - collaborative_weight) * content_score
        score = score.drop(index=seeds, errors="ignore")

        result = self.catalog.set_index("movie_id").copy()
        result = result[result["rating_count"] >= min_rating_count]
        result["score"] = score.reindex(result.index)
        result = result.dropna(subset=["score"]).sort_values(
            ["score", "average_rating", "rating_count"],
            ascending=[False, False, False],
        )
        result = result.head(n).reset_index()
        return result[["movie_id", "title", "year", "genres", "average_rating", "rating_count", "score"]]

    def title_for(self, movie_id: int) -> str:
        """Return an unambiguous display label for a movie ID."""
        if self.catalog is None:
            raise RuntimeError("Call fit before reading the catalog.")
        row = self.catalog.loc[self.catalog["movie_id"] == movie_id].iloc[0]
        year = f" ({int(row['year'])})" if pd.notna(row["year"]) else ""
        return f"{row['title']}{year}"

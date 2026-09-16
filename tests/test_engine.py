from __future__ import annotations

import pandas as pd
import pytest

from movie_recommender import GENRE_COLUMNS, MovieRecommender, load_movielens


def movie(movie_id: int, title: str, genres: list[str]) -> dict:
    values = {column: int(column in genres) for column in GENRE_COLUMNS}
    return {
        "movie_id": movie_id,
        "title": title,
        "year": 2000 + movie_id,
        "genres": ", ".join(genres),
        **values,
    }


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.DataFrame(
        [
            (1, 1, 5), (1, 2, 4), (1, 3, 1),
            (2, 1, 4), (2, 2, 5), (2, 3, 1),
            (3, 1, 5), (3, 2, 4), (3, 4, 2),
            (4, 1, 1), (4, 3, 5), (4, 4, 4),
        ],
        columns=["user_id", "movie_id", "rating"],
    )
    movies = pd.DataFrame(
        [
            movie(1, "Alpha", ["Action"]),
            movie(2, "Bravo", ["Action"]),
            movie(3, "Charlie", ["Drama"]),
            movie(4, "Delta", ["Drama"]),
        ]
    )
    return ratings, movies


def test_recommendations_exclude_seed_and_are_ranked(sample_data) -> None:
    ratings, movies = sample_data
    model = MovieRecommender(shrinkage=1).fit(ratings, movies)

    result = model.recommend([1], n=3, collaborative_weight=1, min_rating_count=1)

    assert 1 not in result["movie_id"].tolist()
    assert result["score"].is_monotonic_decreasing
    assert result.iloc[0]["movie_id"] == 2


def test_multiple_seeds_are_deduplicated_and_excluded(sample_data) -> None:
    ratings, movies = sample_data
    model = MovieRecommender().fit(ratings, movies)

    result = model.recommend([1, 1, 2], n=10, min_rating_count=1)

    assert set(result["movie_id"]).isdisjoint({1, 2})


def test_minimum_rating_filter_is_applied(sample_data) -> None:
    ratings, movies = sample_data
    model = MovieRecommender().fit(ratings, movies)

    result = model.recommend([1], n=10, min_rating_count=3)

    assert (result["rating_count"] >= 3).all()


def test_invalid_requests_raise_clear_errors(sample_data) -> None:
    ratings, movies = sample_data
    model = MovieRecommender().fit(ratings, movies)

    with pytest.raises(ValueError, match="at least one"):
        model.recommend([])
    with pytest.raises(KeyError, match="Unknown movie"):
        model.recommend([999])
    with pytest.raises(ValueError, match="between 0 and 1"):
        model.recommend([1], collaborative_weight=1.5)


def test_loader_reports_missing_files(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="u.data"):
        load_movielens(tmp_path)

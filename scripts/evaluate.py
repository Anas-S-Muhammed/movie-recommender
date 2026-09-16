"""Run a reproducible leave-one-out ranking evaluation."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movie_recommender import MovieRecommender, load_movielens  # noqa: E402


def evaluate(k: int, users: int, seed: int) -> dict[str, float | int]:
    ratings, movies = load_movielens(ROOT / "data")
    positive = ratings[ratings["rating"] >= 4]
    eligible = positive.groupby("user_id").filter(lambda frame: len(frame) >= 3)
    user_ids = sorted(eligible["user_id"].unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(user_ids)
    user_ids = user_ids[: min(users, len(user_ids))]

    held_out: dict[int, int] = {}
    for user_id in user_ids:
        candidates = eligible.loc[eligible["user_id"] == user_id, "movie_id"].tolist()
        held_out[user_id] = rng.choice(candidates)

    held_out_pairs = {(user_id, movie_id) for user_id, movie_id in held_out.items()}
    train_mask = [
        (user_id, movie_id) not in held_out_pairs
        for user_id, movie_id in zip(ratings["user_id"], ratings["movie_id"], strict=True)
    ]
    train = ratings.loc[train_mask]
    model = MovieRecommender(shrinkage=25).fit(train, movies)

    hits = 0
    reciprocal_rank = 0.0
    evaluated = 0
    for user_id, target in held_out.items():
        seeds = train.loc[(train["user_id"] == user_id) & (train["rating"] >= 4), "movie_id"].tolist()
        seeds = [movie_id for movie_id in seeds if movie_id in model.collaborative_similarity.columns]
        if not seeds or target not in model.collaborative_similarity.columns:
            continue
        result = model.recommend(seeds[-10:], n=k, min_rating_count=5)
        ranked = result["movie_id"].tolist()
        evaluated += 1
        if target in ranked:
            rank = ranked.index(target) + 1
            hits += 1
            reciprocal_rank += 1 / rank

    return {
        "k": k,
        "requested_users": users,
        "evaluated_users": evaluated,
        "hit_rate_at_k": hits / evaluated if evaluated else 0.0,
        "mean_reciprocal_rank": reciprocal_rank / evaluated if evaluated else 0.0,
        "random_seed": seed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--users", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.k, args.users, args.seed), indent=2))

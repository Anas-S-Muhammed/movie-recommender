# Movie Recommender

Item-based collaborative filtering recommender built on the MovieLens 100K dataset, served through a Streamlit UI.

Pick a movie, get back the N most similar titles ranked by cosine similarity over user rating patterns. No content metadata (genre, cast, plot) is used — the similarity is purely behavioral.

## How it works

1. Load `u.data` (100,000 ratings, 943 users) and `u.item` (1,682 movies), merge on `movie_id`.
2. Pivot into a 943 × 1,682 user-movie ratings matrix.
3. Fill missing ratings with 0 and compute cosine similarity between every pair of movie columns (transposed matrix), producing a 1,682 × 1,682 movie-movie similarity matrix.
4. For a selected movie, sort its similarity row and return the top N (excluding itself).

The whole pipeline is wrapped in `@st.cache_data`, so the matrix is built once per session instead of on every interaction.

This is memory-based CF, not model-based — there's no matrix factorization, no learned latent factors, just direct similarity on raw rating vectors.

## Tech stack

| Tool | Role |
|---|---|
| Python | Language |
| Pandas | Loading, merging, pivoting the ratings data |
| scikit-learn | `cosine_similarity` for the movie-movie similarity matrix |
| Streamlit | UI layer and app hosting |

That's the full runtime dependency list (`requirements.txt`). The notebook (`movie_recommender.ipynb`) additionally pulls in NumPy and Matplotlib for exploratory analysis — those aren't dependencies of the app itself.

## Project structure

```
movie-recommender/
├── ml-100k/
│   ├── app.py                    # Streamlit app — the actual product
│   ├── movie_recommender.ipynb   # Exploratory notebook, not required to run the app
│   ├── u.data                    # 100K ratings (user_id, movie_id, rating, timestamp)
│   └── u.item                    # Movie metadata (id, title, genres, etc.)
├── requirements.txt
└── README.md
```

## Running it locally

```bash
git clone https://github.com/Anas-S-Muhammed/movie-recommender.git
cd movie-recommender
pip install -r requirements.txt
streamlit run ml-100k/app.py
```

Opens at `localhost:8501`.

## Live demo

Not currently deployed — the placeholder link from the old README pointed nowhere. If you deploy this on Streamlit Community Cloud, drop the real URL here.

## Dataset

MovieLens 100K, from GroupLens Research (University of Minnesota). 100,000 ratings on a 1–5 scale, 943 users, 1,682 movies, each user has rated at least 20 movies. Standard benchmark dataset for recommender systems, non-commercial use license from GroupLens (see their [dataset page](https://grouplens.org/datasets/movielens/100k/) for terms).

## Known limitations

- **Zero-imputation bias**: filling unrated entries with 0 rather than something like the movie's mean rating skews similarity toward popular movies with lots of ratings, since sparser vectors look more "different" by default.
- **No evaluation**: there's no train/test split or offline metric (RMSE, precision@k) run against this — recommendation quality is eyeballed, not measured.
- **Cold start**: new movies or users with no rating history can't be recommended or given recommendations, since the whole approach depends on an existing rating matrix.
- **O(n²) similarity matrix**: fine at 1,682 movies, won't hold up at catalog sizes past the tens of thousands without approximate nearest-neighbor techniques (e.g. `annoy`, `faiss`) or a switch to model-based CF.

## Possible improvements

- Swap zero-fill for mean-centered ratings or an explicit missing-value mask before computing similarity.
- Add an offline eval harness (train/test split, precision@k or RMSE) so changes to the algorithm can be compared against a baseline instead of judged by feel.
- Move from memory-based CF to matrix factorization (SVD, ALS) for better scaling and to handle sparsity properly.
- Hybrid with content-based signals (genre, release year) to soften the cold-start problem.

## License

No license file is currently in this repo, so by default all rights are reserved and the code isn't legally reusable by others. Add a `LICENSE` file (MIT is the standard choice for a portfolio project like this) if you want it to actually be open source. The MovieLens dataset itself has its own non-commercial license from GroupLens, separate from whatever you pick for the code.

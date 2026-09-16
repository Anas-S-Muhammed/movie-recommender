import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from movie_recommender import MovieRecommender, load_movielens  # noqa: E402

st.set_page_config(page_title="ReelMatch | Movie Recommender", page_icon="🎬", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1150px; padding-top: 2.5rem;}
    [data-testid="stMetric"] {background: #151922; border: 1px solid #2a3040; padding: 1rem; border-radius: 12px;}
    .subtitle {color: #9aa4b2; font-size: 1.08rem; margin-bottom: 1.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Building the recommendation model…")
def build_recommender() -> tuple[MovieRecommender, pd.DataFrame, pd.DataFrame]:
    ratings, movies = load_movielens(ROOT / "data")
    model = MovieRecommender(shrinkage=25).fit(ratings, movies)
    return model, ratings, movies


try:
    model, ratings, movies = build_recommender()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"The application could not load the MovieLens data: {exc}")
    st.stop()

catalog = model.catalog
assert catalog is not None

st.title("🎬 ReelMatch")
st.markdown(
    '<p class="subtitle">Discover movies through audience taste and genre similarity—not sponsored rankings.</p>',
    unsafe_allow_html=True,
)

metric_1, metric_2, metric_3 = st.columns(3)
metric_1.metric("Ratings analyzed", f"{len(ratings):,}")
metric_2.metric("Movies", f"{len(catalog):,}")
metric_3.metric("Users", f"{ratings['user_id'].nunique():,}")

st.divider()

with st.sidebar:
    st.header("Recommendation settings")
    num_recommendations = st.slider("Number of results", 5, 20, 10)
    collaborative_weight = st.slider(
        "Audience-taste weight",
        min_value=0,
        max_value=100,
        value=85,
        step=5,
        help="Higher values prioritize shared rating patterns. Lower values prioritize matching genres.",
    )
    min_rating_count = st.slider(
        "Minimum ratings per movie",
        min_value=1,
        max_value=100,
        value=10,
        help="Increase this to remove recommendations based on very little audience data.",
    )
    st.caption("Model: adjusted-cosine collaborative filtering with significance weighting and genre blending.")

st.subheader("Choose movies you already enjoy")
selected_ids = st.multiselect(
    "Search by title",
    options=catalog["movie_id"].tolist(),
    format_func=model.title_for,
    max_selections=5,
    placeholder="Select up to five movies",
    label_visibility="collapsed",
)

button_col, hint_col = st.columns([1, 3], vertical_alignment="center")
with button_col:
    generate = st.button("Find my movies", type="primary", width="stretch")
with hint_col:
    st.caption("Selecting more than one movie creates a broader taste profile.")

if generate:
    if not selected_ids:
        st.warning("Select at least one movie to get recommendations.")
    else:
        recommendations = model.recommend(
            selected_ids,
            n=num_recommendations,
            collaborative_weight=collaborative_weight / 100,
            min_rating_count=min_rating_count,
        )
        if recommendations.empty:
            st.info("No movies matched these filters. Try lowering the minimum-ratings setting.")
        else:
            display = recommendations.copy()
            display.insert(0, "Rank", range(1, len(display) + 1))
            display["Year"] = display["year"].astype("Int64").astype("string").fillna("—")
            display["Audience rating"] = display["average_rating"].map(lambda value: f"{value:.2f} / 5")
            display["Match"] = display["score"].map(lambda value: f"{max(value, 0) * 100:.1f}%")
            display = display.rename(columns={"title": "Movie", "genres": "Genres", "rating_count": "Ratings"})
            st.subheader("Recommended for you")
            st.dataframe(
                display[["Rank", "Movie", "Year", "Genres", "Audience rating", "Ratings", "Match"]],
                hide_index=True,
                width="stretch",
                column_config={
                    "Rank": st.column_config.NumberColumn(width="small"),
                    "Movie": st.column_config.TextColumn(width="large"),
                    "Genres": st.column_config.TextColumn(width="large"),
                },
            )

with st.expander("How ReelMatch generates recommendations"):
    st.markdown(
        """
        ReelMatch adjusts ratings around each user's personal average, compares movies using cosine
        similarity, and discounts matches supported by very few shared viewers. It then blends that
        audience signal with genre similarity. The system never uses personal information.

        This demonstration uses MovieLens 100K, a historical research dataset, so its catalogue is
        intentionally limited to films available when the dataset was created.
        """
    )

st.divider()
st.caption("Built with Python, scikit-learn and Streamlit · Data: MovieLens 100K by GroupLens Research")

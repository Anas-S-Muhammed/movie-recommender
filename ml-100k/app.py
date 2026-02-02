import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# Load data with caching so it doesn't reload every time
@st.cache_data
def load_data():
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', 
                         names=['movie_id', 'title', 'release_date', 'video_release', 'imdb_url'] + [f'genre_{i}' for i in range(19)],
                         usecols=['movie_id', 'title'])
    
    data = ratings.merge(movies, on='movie_id')
    user_movie_matrix = data.pivot_table(index='user_id', columns='title', values='rating')
    user_movie_matrix_filled = user_movie_matrix.fillna(0)
    movie_similarity = cosine_similarity(user_movie_matrix_filled.T)
    movie_similarity_df = pd.DataFrame(movie_similarity, 
                                        index=user_movie_matrix.columns,
                                        columns=user_movie_matrix.columns)
    return movie_similarity_df, user_movie_matrix.columns.tolist()

movie_similarity_df, movie_list = load_data()

def get_recommendations(movie_title, top_n=10):
    similar_scores = movie_similarity_df[movie_title].sort_values(ascending=False)
    return similar_scores[1:top_n+1]

# UI
st.title("🎬 Movie Recommendation System")
st.markdown("### Find your next favorite movie!")

# Sidebar
st.sidebar.header("Settings")
num_recommendations = st.sidebar.slider("Number of recommendations:", 5, 20, 10)
st.sidebar.markdown("---")
st.sidebar.info("💡 Select a movie you enjoyed and get personalized recommendations based on user ratings!")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    selected_movie = st.selectbox(
        "🔍 Search for a movie you like:",
        options=movie_list,
        index=None,
        placeholder="Type to search..."
    )

with col2:
    st.write("")  # spacing
    st.write("")  # spacing
    get_recs_button = st.button("🎯 Get Recommendations", type="primary", use_container_width=True)

# Show recommendations
if selected_movie and get_recs_button:
    with st.spinner("Finding similar movies..."):
        recs = get_recommendations(selected_movie, top_n=num_recommendations)
    
    st.success(f"✅ Found {len(recs)} movies similar to **{selected_movie}**")
    
    # Display as a nice dataframe
    recs_df = pd.DataFrame({
        'Movie Title': recs.index,
        'Similarity Score': recs.values
    })
    recs_df['Similarity Score'] = recs_df['Similarity Score'].apply(lambda x: f"{x:.3f}")
    recs_df.index = range(1, len(recs_df) + 1)
    
    st.dataframe(recs_df, use_container_width=True, height=400)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Data: MovieLens 100K")
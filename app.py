import streamlit as st
import pickle
import pandas as pd
import requests

# ===================== Function to Fetch Poster from TMDB =====================
def fetch_poster(movie_id):
    """
    Given a TMDB movie ID, fetch the full poster image URL using TMDB API.
    
    Args:
        movie_id (int): The TMDB ID of the movie.
    
    Returns:
        str: Full URL to the movie poster image.
    """
    api_key = "e6ba2f139288bd08eac320a78add299a"  # 🔁 Replace this with your actual TMDB API key
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"

    response = requests.get(url)
    data = response.json()

    poster_path = data.get('poster_path')
    if not poster_path:
        return "https://via.placeholder.com/500x750?text=No+Image"
    
    full_poster_url = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_poster_url

# ===================== Load Data =====================
# Load the movie dataframe (contains 'title' and 'movie_id' columns)
movies_df = pickle.load(open('movies.pkl', 'rb'))

# Load similarity matrix
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Get just the movie titles for the dropdown
movie_list = movies_df['title'].values

# ===================== Recommend Function =====================
def recommend(movie):
    """
    Given a movie title, recommend 5 similar movies with their posters.
    """
    movie_index = movies_df[movies_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_indexes = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movie_indexes:
        movie_id = movies_df.iloc[i[0]].movie_id
        recommended_movies.append(movies_df.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_posters

# ===================== Streamlit UI =====================
st.title('🎬 Movie Recommender System')

selected_movie = st.selectbox("Select a movie to get recommendations", movie_list)

if st.button("Recommend"):
    names, posters = recommend(selected_movie)

    # Display in a horizontal row using Streamlit columns
    cols = st.columns(5)
    for idx, col in enumerate(cols):
        with col:
            st.text(names[idx])
            st.image(posters[idx])

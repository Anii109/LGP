
import streamlit as st
import pickle
import pandas as pd
import requests
import os
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

# ===================== TMDB Poster API =====================
def fetch_poster(movie_id):
    """
    Given a TMDB movie ID, fetch the full poster image URL using TMDB API.
    """
    api_key = "e6ba2f139288bd08eac320a78add299a"  # 🔁 Replace with your own API key
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    
    response = requests.get(url)
    data = response.json()
    poster_path = data.get('poster_path')
    
    if not poster_path:
        return "https://via.placeholder.com/500x750?text=No+Image"
    
    return "https://image.tmdb.org/t/p/w500/" + poster_path

# ===================== Data Processing & Caching =====================
@st.cache_data
def load_and_prepare_data():
    """
    Reads CSV files, processes data, vectorizes tags, and generates similarity matrix.
    Saves the processed data into .pkl files for faster reuse.
    """
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies.dropna(inplace=True)

    def convert(obj):
        return [i['name'] for i in ast.literal_eval(obj)]

    def convert3(obj):
        return [i['name'] for i in ast.literal_eval(obj)[:3]]

    def fetch_director(obj):
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                return [i['name']]
        return []

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    for feature in ['genres', 'keywords', 'cast', 'crew']:
        movies[feature] = movies[feature].apply(lambda x: [i.replace(" ", "") for i in x])

    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    new_df = movies[['movie_id', 'title', 'tags']]
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

    ps = PorterStemmer()
    def stem(text):
        return " ".join([ps.stem(i) for i in text.split()])

    new_df['tags'] = new_df['tags'].apply(stem)

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    similarity = cosine_similarity(vectors)

    pickle.dump(new_df, open('movies.pkl', 'wb'))
    pickle.dump(similarity, open('similarity.pkl', 'wb'))

    return new_df, similarity

# ===================== Load or Regenerate =====================
if not os.path.exists('movies.pkl') or not os.path.exists('similarity.pkl'):
    movies_df, similarity = load_and_prepare_data()
else:
    movies_df = pickle.load(open('movies.pkl', 'rb'))
    similarity = pickle.load(open('similarity.pkl', 'rb'))

# ===================== Recommendation Logic =====================
def recommend(movie):
    """
    Given a movie title, returns 5 most similar movies and their poster URLs.
    """
    movie = movie.lower()
    try:
        movie_index = movies_df[movies_df['title'].str.lower() == movie].index[0]
    except IndexError:
        return [], []

    distances = similarity[movie_index]
    movie_indexes = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movie_indexes:
        movie_id = movies_df.iloc[i[0]].movie_id
        recommended_movies.append(movies_df.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_posters

# ===================== Streamlit Web App =====================
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title('🎬 Movie Recommender System')

selected_movie = st.selectbox("Search for a movie to get recommendations", movies_df['title'].values)

if st.button("Recommend"):
    names, posters = recommend(selected_movie)
    
    if names:
        st.subheader("Top 5 Recommended Movies:")
        cols = st.columns(5)
        for idx, col in enumerate(cols):
            with col:
                st.text(names[idx])
                st.image(posters[idx])
    else:
        st.error("Movie not found or insufficient data.")

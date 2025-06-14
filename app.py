'''
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

import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import os
import ast
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =================== TMDB API KEY ===================
API_KEY = "e6ba2f139288bd08eac320a78add299a"  # Replace with your TMDB key

# =================== Fetch Poster from TMDB ===================
def fetch_movie_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    response = requests.get(url)
    data = response.json()
    return {
        "poster": "https://image.tmdb.org/t/p/w500/" + data.get('poster_path') if data.get('poster_path') else None,
        "title": data.get("title"),
        "genres": ", ".join([genre['name'] for genre in data.get("genres", [])]),
        "rating": data.get("vote_average"),
        "overview": data.get("overview"),
        "release_date": data.get("release_date")
    }

# =================== Generate Pickle Files ===================
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

def fetch_director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return [i['name']]
    return []

def stem(text):
    ps = PorterStemmer()
    return " ".join([ps.stem(i) for i in text.split()])

def generate_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies.dropna(inplace=True)

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    for feature in ['genres', 'keywords', 'cast', 'crew']:
        movies[feature] = movies[feature].apply(lambda x: [i.replace(" ", "") for i in x])

    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    new_df = movies[['movie_id', 'title', 'tags']]
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
    new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
    new_df['tags'] = new_df['tags'].apply(stem)

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()

    similarity = cosine_similarity(vectors)

    pickle.dump(new_df, open('movies.pkl', 'wb'))
    pickle.dump(similarity, open('similarity.pkl', 'wb'))

# =================== Recommend Function ===================
def recommend(movie):
    movie_index = movies_df[movies_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended = []
    for i in movie_list:
        movie_id = movies_df.iloc[i[0]].movie_id
        details = fetch_movie_details(movie_id)
        recommended.append(details)
    return recommended

# =================== Main App ===================
if not os.path.exists("movies.pkl") or not os.path.exists("similarity.pkl"):
    generate_data()

movies_df = pickle.load(open("movies.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))

st.title("🎬 Movie Recommender System")

selected_movie = st.selectbox("Select a movie to get recommendations", movies_df['title'].values)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    for movie in recommendations:
        st.subheader(movie["title"])
        cols = st.columns([1, 2])
        with cols[0]:
            if movie["poster"]:
                st.image(movie["poster"])
            else:
                st.text("No Image")
        with cols[1]:
            st.markdown(f"**Genres:** {movie['genres']}")
            st.markdown(f"**Rating:** {movie['rating']}")
            st.markdown(f"**Release Date:** {movie['release_date']}")
            st.markdown(f"**Overview:** {movie['overview']}")
        st.markdown("---")
'''
import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import os
import ast
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =================== TMDB API KEY ===================
API_KEY = "e6ba2f139288bd08eac320a78add299a"  # Replace with your TMDB key

# =================== Fetch Poster and Details ===================
def fetch_movie_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    response = requests.get(url)
    data = response.json()
    return {
        "poster": "https://image.tmdb.org/t/p/w500/" + data.get('poster_path') if data.get('poster_path') else None,
        "title": data.get("title"),
        "genres": ", ".join([genre['name'] for genre in data.get("genres", [])]),
        "rating": data.get("vote_average"),
        "overview": data.get("overview"),
        "release_date": data.get("release_date"),
        "id": movie_id  # include ID to fetch trailer
    }

# =================== Fetch Trailer Link ===================
def fetch_movie_trailer(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={API_KEY}&language=en-US"
    response = requests.get(url)
    data = response.json()
    for video in data.get("results", []):
        if video["site"] == "YouTube" and video["type"] == "Trailer":
            return f"https://www.youtube.com/watch?v={video['key']}"
    return None

# =================== Generate Pickle Files ===================
def convert(obj):
    return [i['name'] for i in ast.literal_eval(obj)]

def convert3(obj):
    return [i['name'] for i in ast.literal_eval(obj)[:3]]

def fetch_director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return [i['name']]
    return []

def stem(text):
    ps = PorterStemmer()
    return " ".join([ps.stem(i) for i in text.split()])

def generate_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies.dropna(inplace=True)

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    for feature in ['genres', 'keywords', 'cast', 'crew']:
        movies[feature] = movies[feature].apply(lambda x: [i.replace(" ", "") for i in x])

    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    new_df = movies[['movie_id', 'title', 'tags']]
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
    new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
    new_df['tags'] = new_df['tags'].apply(stem)

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()

    similarity = cosine_similarity(vectors)

    pickle.dump(new_df, open('movies.pkl', 'wb'))
    pickle.dump(similarity, open('similarity.pkl', 'wb'))

# =================== Recommend Function ===================
def recommend(movie):
    movie_index = movies_df[movies_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended = []
    for i in movie_list:
        movie_id = movies_df.iloc[i[0]].movie_id
        details = fetch_movie_details(movie_id)
        trailer_url = fetch_movie_trailer(movie_id)
        details['trailer_url'] = trailer_url
        recommended.append(details)
    return recommended

# =================== Main App ===================
if not os.path.exists("movies.pkl") or not os.path.exists("similarity.pkl"):
    generate_data()

movies_df = pickle.load(open("movies.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))

st.title("🎬 Movie Recommender System")

selected_movie = st.selectbox("Select a movie to get recommendations", movies_df['title'].values)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    for movie in recommendations:
        st.subheader(movie["title"])
        cols = st.columns([1, 2])
        with cols[0]:
            if movie["poster"]:
                st.image(movie["poster"])
            else:
                st.text("No Image")
        with cols[1]:
            st.markdown(f"**Genres:** {movie['genres']}")
            st.markdown(f"**Rating:** {movie['rating']}")
            st.markdown(f"**Release Date:** {movie['release_date']}")
            st.markdown(f"**Overview:** {movie['overview']}")
            if movie['trailer_url']:
               trailer_button= st.markdown(f"[▶️ Watch Trailer]({movie['trailer_url']})", unsafe_allow_html=True)
            else:
                st.markdown("🚫 Trailer not available")
        st.markdown("---")
        '''
import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import os
import ast
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =================== TMDB API KEY ===================
API_KEY = "e6ba2f139288bd08eac320a78add299a"  # Replace with your TMDB key

# =================== Fetch Poster and Details ===================
def fetch_movie_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    response = requests.get(url)
    data = response.json()
    return {
        "poster": "https://image.tmdb.org/t/p/w500/" + data.get('poster_path') if data.get('poster_path') else None,
        "title": data.get("title"),
        "genres": ", ".join([genre['name'] for genre in data.get("genres", [])]),
        "rating": data.get("vote_average"),
        "overview": data.get("overview"),
        "release_date": data.get("release_date"),
        "id": movie_id  # include ID to fetch trailer
    }

# =================== Fetch Trailer Link ===================
def fetch_movie_trailer(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={API_KEY}&language=en-US"
    response = requests.get(url)
    data = response.json()
    for video in data.get("results", []):
        if video["site"] == "YouTube" and video["type"] == "Trailer":
            return f"https://www.youtube.com/watch?v={video['key']}"
    return None

# =================== Generate Pickle Files ===================
def convert(obj):
    return [i['name'] for i in ast.literal_eval(obj)]

def convert3(obj):
    return [i['name'] for i in ast.literal_eval(obj)[:3]]

def fetch_director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return [i['name']]
    return []

def stem(text):
    ps = PorterStemmer()
    return " ".join([ps.stem(i) for i in text.split()])

def generate_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies.dropna(inplace=True)

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    for feature in ['genres', 'keywords', 'cast', 'crew']:
        movies[feature] = movies[feature].apply(lambda x: [i.replace(" ", "") for i in x])

    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    new_df = movies[['movie_id', 'title', 'tags']]
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
    new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
    new_df['tags'] = new_df['tags'].apply(stem)

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()

    similarity = cosine_similarity(vectors)

    pickle.dump(new_df, open('movies.pkl', 'wb'))
    pickle.dump(similarity, open('similarity.pkl', 'wb'))

# =================== Recommend Function ===================
def recommend(movie):
    movie_index = movies_df[movies_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended = []
    for i in movie_list:
        movie_id = movies_df.iloc[i[0]].movie_id
        details = fetch_movie_details(movie_id)
        trailer_url = fetch_movie_trailer(movie_id)
        details['trailer_url'] = trailer_url
        recommended.append(details)
    return recommended

# =================== Main App ===================
if not os.path.exists("movies.pkl") or not os.path.exists("similarity.pkl"):
    generate_data()

movies_df = pickle.load(open("movies.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))

st.title("🎬 Movie Recommender System")

selected_movie = st.selectbox("Select a movie to get recommendations", movies_df['title'].values)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    for movie in recommendations:
        st.subheader(movie["title"])
        cols = st.columns([1, 2])
        with cols[0]:
            if movie["poster"]:
                st.image(movie["poster"])
            else:
                st.text("No Image")
        with cols[1]:
            st.markdown(f"**Genres:** {movie['genres']}")
            st.markdown(f"**Rating:** {movie['rating']}")
            st.markdown(f"**Release Date:** {movie['release_date']}")
            st.markdown(f"**Overview:** {movie['overview']}")
            if movie['trailer_url']:
                trailer_button = st.button(f"▶️ Watch Trailer: {movie['title']}", key=movie['id'])
                if trailer_button:
                    st.markdown(f"<meta http-equiv='refresh' content='0; url={movie['trailer_url']}' />", unsafe_allow_html=True)
            else:
                st.markdown("🚫 Trailer not available")
        st.markdown("---")
'''
'''
import streamlit as st
import pandas as pd
import requests
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------- TMDB API Setup ---------------------
API_KEY = "e6ba2f139288bd08eac320a78add299a"  # 🔁 Replace with your API key

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    response = requests.get(url)
    data = response.json()
    poster_path = data.get('poster_path')
    if poster_path:
        return "https://image.tmdb.org/t/p/w500/" + poster_path
    return "https://via.placeholder.com/500x750?text=No+Image"

def fetch_trailer_url(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={API_KEY}&language=en-US"
    response = requests.get(url)
    data = response.json()
    videos = data.get('results', [])
    for video in videos:
        if video['site'] == 'YouTube' and video['type'] == 'Trailer':
            return f"https://www.youtube.com/watch?v={video['key']}"
    return None

# --------------------- Generate Data ---------------------
def generate_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    movies = movies.merge(credits, on='title')

    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies.dropna(inplace=True)

    import ast

    def convert(obj):
        return [i['name'].replace(" ", "") for i in ast.literal_eval(obj)]

    def convert3(obj):
        return [i['name'].replace(" ", "") for i in ast.literal_eval(obj)[:3]]

    def fetch_director(obj):
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                return [i['name'].replace(" ", "")]
        return []

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    new_df = movies[['movie_id', 'title', 'tags']]
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

    from nltk.stem.porter import PorterStemmer
    ps = PorterStemmer()

    def stem(text):
        return " ".join([ps.stem(word) for word in text.split()])

    new_df['tags'] = new_df['tags'].apply(stem)

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    similarity = cosine_similarity(vectors)

    return new_df, similarity

# --------------------- Load or Generate Data ---------------------
@st.cache(allow_output_mutation=True)
def load_data():
    return generate_data()

movies_df, similarity = load_data()

# --------------------- Recommend Function ---------------------
def recommend(movie):
    index = movies_df[movies_df['title'] == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommendations = []
    for i in movie_list:
        movie_id = movies_df.iloc[i[0]].movie_id
        title = movies_df.iloc[i[0]].title
        poster = fetch_poster(movie_id)
        trailer = fetch_trailer_url(movie_id)
        recommendations.append((title, poster, trailer))
    return recommendations

# --------------------- Streamlit UI ---------------------
st.set_page_config(layout="wide")
st.title("🎬 Movie Recommender System")

selected_movie = st.selectbox("Select a movie to get recommendations", movies_df['title'].values)

if st.button("Recommend"):
    results = recommend(selected_movie)

    cols = st.columns(5)
    for idx, col in enumerate(cols):
        with col:
            title, poster_url, trailer_url = results[idx]
            st.image(poster_url, use_column_width=True)
            st.markdown(f"**{title}**")
            if trailer_url:
                st.markdown(f"""
                    <a href="{trailer_url}" target="_blank">
                        <button style="background-color:#ff4b4b;color:white;border:none;padding:8px 16px;border-radius:5px;cursor:pointer;">
                            ▶️ Watch Trailer
                        </button>
                    </a>
                """, unsafe_allow_html=True)
            else:
                st.caption("Trailer not available.")
'''
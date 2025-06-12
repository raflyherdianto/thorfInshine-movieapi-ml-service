import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

# Path ini memastikan file akan ditemukan dengan benar di server Azure
_current_dir = os.path.dirname(os.path.abspath(__file__))
_movies_content_path = os.path.join(_current_dir, 'models', 'movies_content.csv')
_movie_id_mappings_path = os.path.join(_current_dir, 'models', 'movie_id_mappings.json')

# Muat data
try:
    movies_content_df = pd.read_csv(_movies_content_path)
    with open(_movie_id_mappings_path, 'r') as f:
        movie_id_mappings = json.load(f)
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    raise

# --- PERBAIKAN 1: Menggunakan kolom 'content_features' ---
# Mengganti 'overview' dengan 'content_features' dan mengisi nilai yang kosong.
movies_content_df['content_features'] = movies_content_df['content_features'].fillna('')

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# --- PERBAIKAN 2: Menggunakan kolom 'content_features' untuk fit_transform ---
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_content_df['content_features'])

# Menghitung cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Fungsi untuk mendapatkan rekomendasi
def get_recommendations(user_id):
    # Logika fungsi ini tidak perlu diubah, karena sudah benar
    # dalam mereferensikan movie_id_mappings dan cosine_sim matrix.
    if str(user_id) not in movie_id_mappings:
        raise ValueError(f"User ID '{user_id}' not found.")
        
    movie_id = movie_id_mappings[str(user_id)]
    
    # Mencari index dari film yang cocok dengan ID
    try:
        idx = movies_content_df[movies_content_df['movieId'] == movie_id].index[0]
    except IndexError:
        raise ValueError(f"Movie with ID '{movie_id}' not found in the database.")

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    
    # Mengembalikan judul film yang direkomendasikan
    return movies_content_df['title'].iloc[movie_indices].tolist()

# Fungsi untuk mencari film
def search_movie(title):
    # Mencari judul yang mengandung query (case-insensitive)
    results = movies_content_df[movies_content_df['title'].str.contains(title, case=False, na=False)]
    
    if results.empty:
        return []
        
    # Mengembalikan hasil dalam format list of dictionaries
    return results[['movieId', 'title', 'genres']].to_dict('records')
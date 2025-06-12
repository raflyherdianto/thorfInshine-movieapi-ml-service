import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

# Dapatkan direktori tempat file utils.py ini berada
# Ini memastikan path akan selalu benar, bahkan di server Azure
_current_dir = os.path.dirname(os.path.abspath(__file__))

# Buat path absolut ke file model Anda
_movies_content_path = os.path.join(_current_dir, 'models', 'movies_content.csv')
_movie_id_mappings_path = os.path.join(_current_dir, 'models', 'movie_id_mappings.json')

# Muat data menggunakan path absolut
try:
    movies_content_df = pd.read_csv(_movies_content_path)
    with open(_movie_id_mappings_path, 'r') as f:
        movie_id_mappings = json.load(f)
except FileNotFoundError as e:
    # Tambahkan pesan error yang lebih jelas jika file tidak ditemukan
    print(f"Error loading model files: {e}")
    print(f"Attempted to load from: {_movies_content_path} and {_movie_id_mappings_path}")
    # Anda mungkin ingin menghentikan aplikasi atau menangani ini dengan cara lain
    raise

# --- Sisa kode Anda tetap sama ---

# Preprocessing dan TF-IDF Vectorization
movies_content_df['overview'] = movies_content_df['overview'].fillna('')
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_content_df['overview'])

# Hitung cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Fungsi untuk mendapatkan rekomendasi
def get_recommendations(movie_id, cosine_sim=cosine_sim):
    try:
        if str(movie_id) not in movie_id_mappings:
            return {"error": "Movie ID not found in mappings"}
        
        idx = movie_id_mappings[str(movie_id)]
        
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11] # Ambil 10 film teratas
        
        movie_indices = [i[0] for i in sim_scores]
        
        # Ambil detail film yang direkomendasikan
        recommended_movies = movies_content_df.iloc[movie_indices][[
            'id', 'title', 'overview', 'release_date', 'poster_path', 'vote_average'
        ]].to_dict('records')
        
        return {"recommendations": recommended_movies}
    except Exception as e:
        # Menangani error yang mungkin terjadi
        return {"error": str(e)}
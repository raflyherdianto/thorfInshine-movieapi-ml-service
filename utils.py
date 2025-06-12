import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import joblib

# Path ini memastikan file akan ditemukan dengan benar di server Azure
_current_dir = os.path.dirname(os.path.abspath(__file__))
_movies_content_path = os.path.join(_current_dir, 'models', 'movies_content.csv')
_movie_id_mappings_path = os.path.join(_current_dir, 'models', 'movie_id_mappings.json')
# Path ke file model yang sudah jadi
_vectorizer_path = os.path.join(_current_dir, 'models', 'tfidf_vectorizer.joblib')
_matrix_path = os.path.join(_current_dir, 'models', 'tfidf_matrix.joblib')

# Muat semua file data dan model
try:
    movies_content_df = pd.read_csv(_movies_content_path)
    with open(_movie_id_mappings_path, 'r') as f:
        movie_id_mappings = json.load(f)
    
    # MUAT model, jangan buat lagi
    tfidf_vectorizer = joblib.load(_vectorizer_path)
    tfidf_matrix = joblib.load(_matrix_path)

except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    raise

# Menghitung cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Fungsi get_recommendations dan search_movie 
def get_recommendations(user_id):
    if str(user_id) not in movie_id_mappings:
        raise ValueError(f"User ID '{user_id}' not found.")
    movie_id = movie_id_mappings[str(user_id)]
    try:
        idx = movies_content_df[movies_content_df['movieId'] == movie_id].index[0]
    except IndexError:
        raise ValueError(f"Movie with ID '{movie_id}' not found in the database.")
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies_content_df['title'].iloc[movie_indices].tolist()

def search_movie(title):
    results = movies_content_df[movies_content_df['title'].str.contains(title, case=False, na=False)]
    if results.empty:
        return []
    return results[['movieId', 'title', 'genres']].to_dict('records')

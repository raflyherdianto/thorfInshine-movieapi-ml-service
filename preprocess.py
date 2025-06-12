import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

print("Mulai pra-pemrosesan...")

# Tentukan path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_csv_path = os.path.join(_current_dir, 'models', 'movies_content.csv')
_model_dir = os.path.join(_current_dir, 'models')

# Pastikan direktori model ada
os.makedirs(_model_dir, exist_ok=True)

# Muat data
print("Memuat movies_content.csv...")
movies_content_df = pd.read_csv(_csv_path)

# Isi nilai kosong
movies_content_df['content_features'] = movies_content_df['content_features'].fillna('')

# Buat dan latih TfidfVectorizer
print("Membuat dan melatih TfidfVectorizer...")
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_content_df['content_features'])

# Simpan objek vectorizer dan matrix ke file menggunakan joblib
vectorizer_path = os.path.join(_model_dir, 'tfidf_vectorizer.joblib')
matrix_path = os.path.join(_model_dir, 'tfidf_matrix.joblib')

print(f"Menyimpan vectorizer ke {vectorizer_path}")
joblib.dump(tfidf_vectorizer, vectorizer_path)

print(f"Menyimpan matrix ke {matrix_path}")
joblib.dump(tfidf_matrix, matrix_path)

print("Pra-pemrosesan selesai!")
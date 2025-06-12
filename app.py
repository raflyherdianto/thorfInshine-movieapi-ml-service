from flask import Flask, request, jsonify
from utils import get_recommendations, search_movie

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# --- Endpoint Utama & Pencarian ---

@app.route("/", methods=['GET'])
def home():
    """
    Endpoint utama.
    Memberikan pesan selamat datang.
    """
    return jsonify({"message": "Welcome to the Movie Recommendation API (Flask Version)!"})

@app.route("/search", methods=['GET'])
def search():
    """
    Endpoint untuk mencari film berdasarkan judul.
    Menerima judul film dari query parameter 'title'.
    Contoh: /search?title=Toy Story
    """
    # Mengambil 'title' dari argumen URL
    movie_title = request.args.get('title')

    if not movie_title:
        return jsonify({"error": "Query parameter 'title' is required."}), 400

    try:
        search_results = search_movie(movie_title)
        return jsonify({"results": search_results})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# --- Endpoint Rekomendasi ---

@app.route("/recommend", methods=['POST'])
def recommend():
    """
    Endpoint untuk mendapatkan rekomendasi film berdasarkan user_id.
    Menerima JSON body dengan format: {"user_id": "some_user_id"}
    """
    # Mendapatkan data JSON dari body request
    data = request.get_json()

    if not data or 'user_id' not in data:
        return jsonify({"error": "JSON body with 'user_id' is required."}), 400

    user_id = data['user_id']

    try:
        recommendations = get_recommendations(user_id)
        return jsonify({
            "user_id": user_id,
            "recommendations": recommendations
        })
    except ValueError as e:
        # Menangani kasus jika user_id tidak ditemukan
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# --- Endpoint Baru untuk Testing di Azure ---

@app.route("/test", methods=['GET'])
def test_get():
    """
    Endpoint GET untuk testing.
    Jika Anda bisa mengakses ini setelah deploy, berarti endpoint GET berfungsi.
    """
    return jsonify({"message": "Test GET endpoint is working successfully!"}), 200

@app.route("/test", methods=['POST'])
def test_post():
    """
    Endpoint POST untuk testing.
    Mengembalikan data yang Anda kirim dalam body JSON.
    """
    try:
        data = request.get_json()
        return jsonify({
            "message": "Test POST endpoint is working successfully!",
            "data_received": data
        }), 200
    except Exception as e:
        return jsonify({"error": f"Could not parse JSON body: {str(e)}"}), 400

# Menjalankan aplikasi jika file ini dieksekusi secara langsung
if __name__ == "__main__":
    app.run(debug=True, port=5000)
from flask import Flask, request, jsonify, send_from_directory
from notebook.recommendation_utils import get_recommendations
import pickle
import pandas as pd
import numpy as np
import os
from fuzzywuzzy import process

app = Flask(__name__)

# Get the path to the current file
current_file_path = os.path.abspath(__file__)
# Get the parent directory of the current file
project_root = os.path.dirname(current_file_path)
processed_data_path = os.path.join(project_root, 'notebook', 'processed_data')
model_path = os.path.join(project_root, 'notebook', 'model')
frontend_path = os.path.join(project_root, 'frontend')

# Load the trained recommendation model
with open(os.path.join(model_path, "movie_recommender_model1.pkl"), "rb") as f:
    model_data = pickle.load(f)
    similarity_matrix = model_data["similarity_matrix"]
    movie_ids = model_data["movie_ids"]

# Load the movie_indices
with open(os.path.join(processed_data_path,'movie_indices1.pkl'), 'rb') as f:
    movie_indices = pickle.load(f)

# Load the movie data to get the title
movies_df = movies_df = pd.read_csv("notebook/processed_data/movies_processed1.csv")
# Convert movie titles to lowercase
movies_df['title_clean_lower'] = movies_df['title_clean'].str.lower()
movie_titles = movies_df['title_clean_lower'].tolist()
movie_id_by_title = movies_df.set_index('title_clean_lower')['movieId'].to_dict()

def get_movie_id_from_title(movie_title, movie_titles, movie_id_by_title):
    """
    Return the movie id of a movie by its title.
    Uses fuzzy matching if an exact match is not found.
    """
    movie_title_lower = movie_title.lower()

    # Check for exact match first
    if movie_title_lower in movie_titles:
        return movie_id_by_title[movie_title_lower]

    # Fuzzy matching
    closest_match, score = process.extractOne(movie_title_lower, movie_titles)

    # If the score is good enough, use the closest match
    if score >= 80:  # Adjust the threshold as needed
        print(f"Fuzzy match found: '{movie_title}' -> '{closest_match}' (score: {score})")
        return movie_id_by_title[closest_match]

    return None  # No match found

@app.route("/recommend", methods=["GET"])
def recommend():
    movie_title = request.args.get("movie")
    if not movie_title:
        return jsonify({"error": "No movie title provided"}), 400
    
    movie_id = get_movie_id_from_title(movie_title, movie_titles, movie_id_by_title)

    if movie_id is None:
        return jsonify({"error": f"No movie found with the title: {movie_title}"}), 404
    
    if movie_id not in movie_indices.keys():
      return jsonify({"error": f"No movie found with the title: {movie_title}"}), 404

    idx = movie_indices.get(movie_id)
    if idx is None or idx >= similarity_matrix.shape[0]:
        return jsonify({"error": f"No valid movie found for: {movie_title}"}), 404

    recommendations_df = get_recommendations([idx], similarity_matrix, movie_ids, movies_df)

    if recommendations_df.empty:
        return jsonify({"message": f"No recommendations found for {movie_title}"}),200

    recommendations = recommendations_df['title_clean'].tolist()

    return jsonify({"recommendations": recommendations})

@app.route("/")
def index():
    return send_from_directory(frontend_path, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(frontend_path, filename)

if __name__ == "__main__":
    app.run(debug=True)

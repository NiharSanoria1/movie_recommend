from flask import Flask, request, jsonify, send_from_directory, render_template, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from notebook.recommendation_utils import get_recommendations
import pickle
import pandas as pd
import numpy as np
import os
from fuzzywuzzy import process

app = Flask(__name__, template_folder="frontend/templates", static_folder="frontend/static")
app.secret_key = 'your_secret_key'  # Change in production

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

# Get the path to the current file
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_file_path)
processed_data_path = os.path.join(project_root, 'notebook', 'processed_data')
model_path = os.path.join(project_root, 'notebook', 'model')
frontend_path = os.path.join(project_root, 'frontend')

# Load the trained recommendation model
with open(os.path.join(model_path, "movie_recommender_model1.pkl"), "rb") as f:
    model_data = pickle.load(f)
    similarity_matrix = model_data["similarity_matrix"]
    movie_ids = model_data["movie_ids"]

# Load the movie indices
with open(os.path.join(processed_data_path, 'movie_indices1.pkl'), 'rb') as f:
    movie_indices = pickle.load(f)

# Load the movie data
movies_df = pd.read_csv(os.path.join(processed_data_path, "movies_processed1.csv"))
movies_df['title_clean_lower'] = movies_df['title_clean'].str.lower()
movie_titles = movies_df['title_clean_lower'].tolist()
movie_id_by_title = movies_df.set_index('title_clean_lower')['movieId'].to_dict()

# Function to get movie ID from title using fuzzy matching
def get_movie_id_from_title(movie_title, movie_titles, movie_id_by_title):
    movie_title_lower = movie_title.lower()
    if movie_title_lower in movie_titles:
        return movie_id_by_title[movie_title_lower]
    closest_match, score = process.extractOne(movie_title_lower, movie_titles)
    if score >= 80:
        return movie_id_by_title[closest_match]
    return None

# Recommendation Route
@app.route("/recommend", methods=["GET"])
def recommend():
    if 'username' not in session:
        return jsonify({"error": "Unauthorized access. Please login."}), 401

    movie_title = request.args.get("movie")
    if not movie_title:
        return jsonify({"error": "No movie title provided"}), 400
    
    movie_id = get_movie_id_from_title(movie_title, movie_titles, movie_id_by_title)
    if movie_id is None or movie_id not in movie_indices:
        return jsonify({"error": f"No movie found with the title: {movie_title}"}), 404
    
    idx = movie_indices.get(movie_id)
    if idx is None or idx >= similarity_matrix.shape[0]:
        return jsonify({"error": f"No valid movie found for: {movie_title}"}), 404

    recommendations_df = get_recommendations([idx], similarity_matrix, movie_ids, movies_df)
    if recommendations_df.empty:
        return jsonify({"message": f"No recommendations found for {movie_title}"}), 200

    recommendations = recommendations_df['title_clean'].tolist()
    return jsonify({"recommendations": recommendations})

# Home Route
@app.route("/")
def home():
    return render_template('index.html', username=session.get('username'))

# Signup Route
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])

        if User.query.filter_by(username=username).first():
            flash("Username already exists! Try logging in.", "error")
            return redirect(url_for("signup"))

        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash("Signup successful! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")

# Login Route
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session["username"] = username
            flash("Login successful!", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password!", "error")

    return render_template("login.html")

# Logout Route
@app.route("/logout")
def logout():
    session.pop("username", None)
    flash("Logged out successfully!", "success")
    return redirect(url_for("home"))

# Static Files Route
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(frontend_path, filename)

# Run App
if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Create database tables
    app.run(debug=True)

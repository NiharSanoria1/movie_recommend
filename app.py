from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load model and data
with open('notebook/model/movie_recommender_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    
similarity_matrix = model_data['similarity_matrix']
movie_indices = model_data['movie_indices'] 
get_recommendations = model_data['get_recommendations']

movies_df = pd.read_csv('notebook/processed_data/movies_light.csv')

from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load model and data
with open('models/movie_recommender_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

similarity_matrix = model_data['similarity_matrix']
movie_indices = model_data['movie_indices'] 
get_recommendations = model_data['get_recommendations']

movies_df = pd.read_csv('processed_data/movies_light.csv')

@app.route('/api/movies/search', methods=['GET'])
def search_movies():
    query = request.args.get('query', '').lower()
    if len(query) < 3:
        return jsonify([])
    
    filtered_movies = movies_df[movies_df['title_clean'].str.lower().str.contains(query)]
    result = filtered_movies.head(10)[['movieId', 'title_clean', 'genres', 'year']].to_dict('records')
    return jsonify(result)

@app.route('/api/recommendations', methods=['POST'])
def recommend_movies():
    data = request.get_json()
    movie_ids = data.get('movie_ids', [])
    
    if not movie_ids:
        return jsonify({'error': 'No movie IDs provided'}), 400
    
    recommendations = get_recommendations(movie_ids, similarity_matrix, movie_indices, 
                                         movies_df, top_n=10)
    
    result = recommendations.to_dict('records')
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
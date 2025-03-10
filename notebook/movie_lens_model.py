from sklearn.metrics.pairwise import cosine_similarity
import pickle
import pandas as pd
import numpy as np

# Load the feature matrix
feature_matrix = np.load('processed_data/feature_matrix.npy')
movie_indices = np.load('processed_data/movie_indices.npy')
movies_df = pd.read_csv('processed_data/movies_processed.csv')

# Calculate the similarity matrix
similarity_matrix = cosine_similarity(feature_matrix)

# Create a function to get recommendations
def get_recommendations(movie_ids, similarity_matrix=similarity_matrix, 
                       movie_indices=movie_indices, movies_df=movies_df, top_n=10):
    """
    Get movie recommendations based on a list of movie IDs
    """
    # Find the indices of the input movies
    movie_idx_list = []
    for movie_id in movie_ids:
        try:
            idx = np.where(movie_indices == movie_id)[0][0]
            movie_idx_list.append(idx)
        except IndexError:
            continue
    
    if not movie_idx_list:
        return []
    
    # Get the average similarity scores across all input movies
    sim_scores = np.zeros(len(similarity_matrix))
    for idx in movie_idx_list:
        sim_scores += similarity_matrix[idx]
    
    sim_scores = sim_scores / len(movie_idx_list)
    
    # Get the indices of movies sorted by similarity score
    sim_scores_with_indices = list(enumerate(sim_scores))
    sim_scores_with_indices = sorted(sim_scores_with_indices, key=lambda x: x[1], reverse=True)
    
    # Filter out the input movies
    sim_scores_with_indices = [x for x in sim_scores_with_indices if x[0] not in movie_idx_list]
    
    # Get the top N most similar movies
    top_movies_indices = [i[0] for i in sim_scores_with_indices[:top_n]]
    
    # Get the movie IDs for the top movies
    top_movie_ids = [movie_indices[i] for i in top_movies_indices]
    
    # Return the top movies' information
    return movies_df[movies_df['movieId'].isin(top_movie_ids)][
        ['movieId', 'title_clean', 'genres', 'year', 'avg_rating', 'num_ratings']
    ].sort_values(by='avg_rating', ascending=False)

# Save the model components
with open('model/movie_recommender_model.pkl', 'wb') as f:
    pickle.dump({
        'similarity_matrix': similarity_matrix,
        'movie_indices': movie_indices,
        'get_recommendations': get_recommendations
    }, f)

print("Recommendation model built and saved.")



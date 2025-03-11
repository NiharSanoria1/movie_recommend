from sklearn.metrics.pairwise import cosine_similarity
from recommendation_utils import get_recommendations
import pickle
import pandas as pd
import numpy as np
import os

# Get the path to the current file
current_file_path = os.path.abspath(__file__)
# Get the parent directory of the current file
parent_directory = os.path.dirname(current_file_path)
# Get the project root directory
project_root = os.path.dirname(parent_directory)
# create the path to the data file
data_folder_path = os.path.join(project_root,"notebook" ,"data", "ml-latest-small")

# Load the data
ratings_df = pd.read_csv(os.path.join(data_folder_path, "ratings.csv"))
movies_df = pd.read_csv(os.path.join(data_folder_path, "movies.csv"))

# Process the movies
# Clean the titles
movies_df["title_clean"] = movies_df["title"].str.extract(r"^(.*?)\s*\(.*$", expand=False)
# remove duplicate
movies_df = movies_df.drop_duplicates(subset='title_clean')

# Keep only the usefull data
movies_df = movies_df.drop(['title','genres'],axis=1)

# aggregate the ratings and keep only the most popular movies
# Group by 'movieId' and count the number of ratings
movie_rating_counts = ratings_df.groupby('movieId')['rating'].count()

# Filter for movies with at least 10 ratings
popular_movies = movie_rating_counts[movie_rating_counts >= 10].index

# Filter the movies_df to keep only popular movies
movies_df = movies_df[movies_df['movieId'].isin(popular_movies)]

# Create the feature matrix

ratings_df = ratings_df[ratings_df['movieId'].isin(movies_df['movieId'])]

# Create a list of the movies_id with the right order
movie_ids = movies_df['movieId'].values
# Create a dictionnary to have the index for each movie id
movie_indices = {movie_id: i for i, movie_id in enumerate(movie_ids)}
# Create the user_movie_matrix with the right order
user_movie_matrix = ratings_df.pivot_table(index='userId', columns='movieId', values='rating')
user_movie_matrix = user_movie_matrix.reindex(columns=movie_ids, fill_value=0)

# Fill NaN values with 0
user_movie_matrix_filled = user_movie_matrix.fillna(0)
# Create the feature matrix
feature_matrix = user_movie_matrix_filled.values

# Calculate the similarity matrix
similarity_matrix = cosine_similarity(feature_matrix.T)
# for debugging
print(f"Feature Matrix Shape: {feature_matrix.shape}")
print(f"Transposed Feature Matrix Shape: {feature_matrix.T.shape}")
print(f"Similarity Matrix Shape: {similarity_matrix.shape}")
print(f"Total movies in dataset: {len(movies_df)}")
print(f"Total movie indices stored: {len(movie_indices)}")

# Save the feature matrix
np.save(os.path.join(project_root,'notebook','processed_data','feature_matrix1.npy'), feature_matrix)

# Save the movie indices
with open(os.path.join(project_root,'notebook','processed_data','movie_indices1.pkl'), 'wb') as f:
    pickle.dump(movie_indices, f)

# save the movie_df
movies_df.to_csv(os.path.join(project_root,'notebook','processed_data','movies_processed1.csv'),index=False)

# Save the model components
with open(os.path.join(project_root,'notebook','model','movie_recommender_model1.pkl'), 'wb') as f:
    pickle.dump({
        'similarity_matrix': similarity_matrix,
        'movie_ids': movie_ids,
    }, f)

print("Recommendation model built and saved.")

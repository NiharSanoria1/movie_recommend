import pandas as pd
import numpy as np
import re
import os

# Create processed_data directory if it doesn't exist
os.makedirs('processed_data', exist_ok=True)

# Load the datasets
# Adjust the file paths based on where you've saved the MovieLens data
movies_df = pd.read_csv('ml-latest-small/movies.csv')
ratings_df = pd.read_csv('ml-latest-small/ratings.csv')
links_df = pd.read_csv('ml-latest-small/links.csv')
tags_df = pd.read_csv('ml-latest-small/tags.csv')

print("Movies dataset shape:", movies_df.shape)
print("Ratings dataset shape:", ratings_df.shape)

# Step 1: Clean and preprocess the movies dataset
def extract_year(title):
    match = re.search(r'\((\d{4})\)$', title)
    if match:
        return int(match.group(1))
    return None

def clean_title(title):
    return re.sub(r'\s*\(\d{4}\)$', '', title).strip()

# Apply the functions to the dataframe
movies_df['year'] = movies_df['title'].apply(extract_year)
movies_df['title_clean'] = movies_df['title'].apply(clean_title)

# Step 2: Process genres
# One-hot encode genres
movies_df['genres_list'] = movies_df['genres'].str.split('|')
all_genres = []
for genres in movies_df['genres_list']:
    all_genres.extend(genres)
unique_genres = sorted(list(set(all_genres)))
if '(no genres listed)' in unique_genres:
    unique_genres.remove('(no genres listed)')

# Create a column for each genre
for genre in unique_genres:
    movies_df[genre] = movies_df['genres'].apply(lambda x: 1 if genre in x else 0)

# Step 3: Process ratings separately and then merge
# Group ratings by movie to get average rating and count
movie_stats = ratings_df.groupby('movieId').agg(
    avg_rating=('rating', 'mean'),
    num_ratings=('rating', 'count')
).reset_index()

# Merge with movies dataframe
# Use left join to keep all movies even if they don't have ratings
movies_with_ratings = pd.merge(movies_df, movie_stats, on='movieId', how='left')

# Fill NaN values for movies with no ratings
movies_with_ratings['avg_rating'] = movies_with_ratings['avg_rating'].fillna(0)
movies_with_ratings['num_ratings'] = movies_with_ratings['num_ratings'].fillna(0)

# Step 4: Create a popularity score (weighted rating)
# This gives preference to movies with more ratings
C = movies_with_ratings['num_ratings'].mean()  # minimum number of votes required
m = movies_with_ratings['avg_rating'].mean()   # mean vote across the whole report

movies_with_ratings['weighted_rating'] = ((movies_with_ratings['num_ratings'] / (movies_with_ratings['num_ratings'] + C)) * 
                               movies_with_ratings['avg_rating']) + ((C / (movies_with_ratings['num_ratings'] + C)) * m)

# Step 5: Merge with links to get external IDs
movies_final = pd.merge(movies_with_ratings, links_df, on='movieId', how='left')

# Convert IDs to proper format
movies_final['imdbId'] = movies_final['imdbId'].fillna(0).astype(int).astype(str).apply(lambda x: f"tt{x.zfill(7)}")
movies_final['tmdbId'] = movies_final['tmdbId'].fillna(0).astype(int)

# Check the final dataframe
print("\nFinal preprocessed dataframe:")
print(movies_final.head())

# Step 6: Save the preprocessed data
movies_final.to_csv('processed_data/movies_processed.csv', index=False)
ratings_df.to_csv('processed_data/ratings.csv', index=False)

# Create a "lightweight" version with just essential columns for the web app
movies_light = movies_final[['movieId', 'title_clean', 'genres', 'year', 'avg_rating', 
                          'num_ratings', 'weighted_rating', 'imdbId', 'tmdbId']]
movies_light.to_csv('processed_data/movies_light.csv', index=False)

print("\nPreprocessed data saved to 'processed_data' directory.")

# Step 7: Create a content matrix for similarity calculation
from sklearn.preprocessing import MinMaxScaler

# Scale numerical features
scaler = MinMaxScaler()
if 'year' in movies_final.columns and movies_final['year'].notna().all():
    movies_final['year_scaled'] = scaler.fit_transform(movies_final[['year']])
else:
    # Handle case where year might be missing for some movies
    movies_final['year_scaled'] = scaler.fit_transform(movies_final[['year']].fillna(movies_final['year'].median()))
    
# Scale ratings
movies_final['rating_scaled'] = scaler.fit_transform(movies_final[['weighted_rating']])

# Prepare the final feature matrix (genres + scaled features)
feature_cols = unique_genres + ['year_scaled', 'rating_scaled']
feature_matrix = movies_final[feature_cols].fillna(0).values

# Save the feature matrix
np.save('processed_data/feature_matrix.npy', feature_matrix)
# Save the corresponding movie IDs
np.save('processed_data/movie_indices.npy', movies_final['movieId'].values)

print("\nFeature matrix created and saved.")
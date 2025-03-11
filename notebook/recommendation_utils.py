import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_recommendations(movie_indexes, similarity_matrix, movie_ids, movies_df):
    """
    Get movie recommendations based on a list of movie index.

    Parameters
    ----------
    movie_indexes : list of int
        List of movie index for which to get recommendations.
    similarity_matrix : numpy.ndarray
        similarity matrix
    movie_ids : numpy.ndarray
        Indices of movies in the feature matrix.
    movies_df : pandas.DataFrame
        DataFrame containing movie information, including titles.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing recommended movies' titles.
    """
    all_recommendations = []

    for idx in movie_indexes:
        sim_scores = similarity_matrix[idx]
        sim_scores_with_indices = list(enumerate(sim_scores))
        sim_scores_with_indices = sorted(sim_scores_with_indices, key=lambda x: x[1], reverse=True)

        recommended_movie_indices = [i for i, score in sim_scores_with_indices[1:6]]
        #get the recommended movie ids
        recommended_movie_ids = [movie_ids[i] for i in recommended_movie_indices]
        
        # Filter out movie IDs that are not in movies_df
        valid_recommended_movie_ids = [movie_id for movie_id in recommended_movie_ids if movie_id in movies_df['movieId'].values]
        recommended_movies = movies_df[movies_df['movieId'].isin(valid_recommended_movie_ids)]
        
        if recommended_movies.empty:
            return pd.DataFrame()  # Return empty DataFrame if no recommendations


        all_recommendations.append(recommended_movies)

    # Concatenate all recommendations into a single DataFrame
    if all_recommendations:
        final_recommendations = pd.concat(all_recommendations)
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no recommendations

    return final_recommendations.drop_duplicates(subset=['movieId'])

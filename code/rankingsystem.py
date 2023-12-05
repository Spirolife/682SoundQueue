import torch
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(encoded_train, encoded_test):
    #flatten the encoded sets
    flat_encoded_train = encoded_train.reshape(encoded_train.shape[0], -1)
    flat_encoded_test = encoded_test.reshape(encoded_test.shape[0], -1)

    #cosine similarity
    similarities = cosine_similarity(flat_encoded_test, flat_encoded_train)
    return similarities

def calculate_genre_similarity(test_genres, train_genres):
    #binary matrix where 1 indicates a genre match, 0 otherwise
    genre_matches = np.array([[genre in test.split('|') for genre in train_genres] for test in test_genres], dtype=int)
    return genre_matches

def calculate_artist_similarity(test_artists, train_artists):
    #binary matrix where 1 indicates an artist match, 0 otherwise
    artist_matches = np.array([[artist in test.split('|') for artist in train_artists] for test in test_artists], dtype=int)
    return artist_matches

def get_ranking(encoded_train, encoded_test, track_df, k=10):
    similarities = calculate_cosine_similarity(encoded_train, encoded_test)

    # Find top k indices for each test song
    top_k_indices = np.argsort(similarities, axis=1)[:, -k:]

    # Get genres and artists for test and train songs
    test_genres = track_df.loc[track_df['sub_split'] == 'test', 'all_genres'].values
    train_genres = track_df.loc[track_df['sub_split'] == 'train', 'all_genres'].values

    test_artists = track_df.loc[track_df['sub_split'] == 'test', 'artist_name'].values
    train_artists = track_df.loc[track_df['sub_split'] == 'train', 'artist_name'].values

    genre_matches = calculate_genre_similarity(test_genres, train_genres)
    artist_matches = calculate_artist_similarity(test_artists, train_artists)

    #initialize scores
    scores = np.zeros((encoded_test.shape[0], encoded_train.shape[0]))

    #assign scores based on cosine similarity, genre, and artist matches
    for i in range(encoded_test.shape[0]):
        for j in range(encoded_train.shape[0]):
            scores[i, j] = similarities[i, j] *0.8 + 0.15 * np.sum(genre_matches[i] * genre_matches[:, j]) + 0.05 * np.sum(artist_matches[i] * artist_matches[:, j])

    # Find top k scores for each test song
    top_k_scores = np.argsort(scores, axis=1)[:, -k:]

    return top_k_scores

if __name__ == "__main__":
    # Use for testing with dummy data

    # Create dummy data
    num_train = 100
    num_test = 10
    encoded_train = torch.randn(num_train - num_test, 128, 32, 32)
    encoded_test = torch.randn(num_test, 128, 32, 32)

    # Create dummy track_df with genres and artists
    track_df = pd.DataFrame(columns=["track_id", "all_genres", "artist_name"])
    track_df["track_id"] = [i for i in range(num_train + num_test)]
    track_df["all_genres"] = ['|'.join(np.random.choice(['Rock', 'Pop', 'Jazz', 'HipHop'], 2)) for _ in range(num_train + num_test)]
    track_df["artist_name"] = ['Artist' + str(np.random.randint(1, 6)) for _ in range(num_train + num_test)]

    top_k_scores = get_ranking(encoded_train, encoded_test, track_df, k=10)

    print("Top K Scores:")
    print(top_k_scores)

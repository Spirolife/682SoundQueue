# PURPOSE OF FILE: Contains the scoring/ranking system for the soundqueue project.

import itertools
import os
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import torch
import utils
from tqdm import tqdm


def reorder_data(index_to_track_id,track_df):
    """Get all rows of test data in track_df and reorder them to match the order of encoded_test using index to track id
    For each test example in encoded_Test, get the corresponding row in track_df and reorder them to match the order of encoded_test.
    
    Args:
        encoded_test (_type_): _description_
        index_to_track_id (_type_): _description_
        track_df (_type_): _description_
    return df of reordered test data
    """
    #Get the track_ids of encoded_test
    test_track_ids = index_to_track_id[index_to_track_id["set_split"] == "test"][["index","track_id"]]

    # make sure they are in index order
    test_track_ids = test_track_ids.sort_values(by=["index"])

    # get the actual rows of the test songs in filtered track_df.
    # the index of the songs are the same as the track_id, get the rows of the track_ids
    test_rows = track_df[track_df[("general","track_id")].isin(test_track_ids["track_id"])]
    
    return test_rows

def get_user_cosine_ranking(encoded_train, encoded_test, track_df, index_to_track_id, k=10):
    # then, use cosine similarity to get a matrix of size (N, 1) where each value is the cosine similarity between the test song and the corresponding training song
    cosine_similarity_model = torch.nn.CosineSimilarity(dim=2, eps=1e-6)

    # get the cosine similarity between each test song and each train song by duplicating the test song N times to match the train set
    big_encoded_test = encoded_test.unsqueeze(0).repeat(encoded_train.shape[0], 1, 1)
    # utils.diagnostic_print("Shape of duplicated encoded_test: {}".format(big_encoded_test.shape))

    # duplicate train set to match the new shape of the test set
    big_encoded_train = encoded_train.unsqueeze(1).repeat(1, encoded_test.shape[0], 1)
    # utils.diagnostic_print("Shape of duplicated encoded_train: {}".format(big_encoded_train.shape))

    # print(encoded_train.shape, encoded_test.shape, big_encoded_train.shape, big_encoded_test.shape)

    # get the cosine similarity between each test song and each train song
    similarities = cosine_similarity_model(big_encoded_test, big_encoded_train).T
    # utils.diagnostic_print("Shape of cosine similarities: {}".format(similarities.shape))

    # print(similarities.shape)

    # for each test item, return the top k train indices
    top_k = torch.topk(similarities, k=k, dim=1)
    utils.diagnostic_print("Shape of top k results for each test: {}".format(top_k.indices.shape))
    # print("Top 1 result for each test: {}".format(top_k.values[:,0], top_k.indices[:,0]))

    top_metadata = []
    for i in range(len(encoded_test)):
        # print(top_k.indices.shape)
        top_k_for_sample = top_k.indices[i,:]
        # these indices refer to the encoded data. use index_to_track_id to get the track ids of the top k results
        top_k_for_sample_track_ids = index_to_track_id.iloc[top_k_for_sample]["track_id"]
        # use track ids to get the metadata of the top k results
        top_k_for_sample_metadata = track_df[track_df[("general","track_id")].isin(top_k_for_sample_track_ids)]
        top_metadata.append(top_k_for_sample_metadata)
        
        # add to the top k results for each test song

    # convert list of dataframes to one dataframe
    top_metadata = pd.concat(top_metadata)

    return top_metadata

    
def get_cosine_ranking(encoded_train, encoded_test, track_df, index_to_track_id, genres_df, k=10):
    print(encoded_train.shape)
    # NOTE: genres_df is for us to visually compare the genres of the top k results with the actual test genres, so we can see how well the model is doing
    # it isnt used for the actual ranking system

    # Use torch cosine similarity: https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html

    # for each test song of shape (255), create a matrix of size (N, 255) where N is the number of songs in the training set
    
    
    # then, use cosine similarity to get a matrix of size (N, 1) where each value is the cosine similarity between the test song and the corresponding training song
    cosine_similarity_model = torch.nn.CosineSimilarity(dim=2, eps=1e-6)

    # get the cosine similarity between each test song and each train song by duplicating the test song N times to match the train set
    big_encoded_test = encoded_test.unsqueeze(0).repeat(encoded_train.shape[0], 1, 1)
    # utils.diagnostic_print("Shape of duplicated encoded_test: {}".format(big_encoded_test.shape))

    # duplicate train set to match the new shape of the test set
    big_encoded_train = encoded_train.unsqueeze(1).repeat(1, encoded_test.shape[0], 1)
    # utils.diagnostic_print("Shape of duplicated encoded_train: {}".format(big_encoded_train.shape))

    # print(encoded_train.shape, encoded_test.shape, big_encoded_train.shape, big_encoded_test.shape)

    # get the cosine similarity between each test song and each train song
    similarities = cosine_similarity_model(big_encoded_test, big_encoded_train).T
    # utils.diagnostic_print("Shape of cosine similarities: {}".format(similarities.shape))

    # print(similarities.shape)

    # for each test item, return the top k train indices
    top_k = torch.topk(similarities, k=k, dim=1)
    utils.diagnostic_print("Shape of top k results for each test: {}".format(top_k.indices.shape))
    # print("Top 1 result for each test: {}".format(top_k.values[:,0], top_k.indices[:,0]))

    #####Evaluation#####

    # print the genres of the actual test songs and the genres of their top match.
    # index_to_track_id has columns: [sub_split, index, track_id] so accessing test songs is index_to_track_id["test", :, "track_id"]
    # then use the track_id to get the genres of the test songs, where track_df has columns: [track_id, ..., all_genres]
    # in track_df, the track_id is NOT the index, so we have to use iloc to get the row of the track_id
    ordered_test_metadata = reorder_data(index_to_track_id,track_df)

    # print(f'Num test examples: {len(test_data)}')
    
    #Ensure the order of the data is correct by printing the index_to_track_id and test_data 
    # print(f'index_to_track_id: {index_to_track_id}')
    # print(f'test_data: {test_data}')
    
    #Print columns of test_data
    print(f'Columns of test_data: {ordered_test_metadata.columns}')
    print(f'Columns of track_df: {track_df.columns}')
    
    test_genres = ordered_test_metadata[[("track","genres_all")]].values     
    test_artists = ordered_test_metadata[[("artist","name")]].values

    # print("Test genres: {}".format(test_genres))
    print("Top match genres for first test sample: SAMPLE {}, PREDICTED {}".format(ordered_test_metadata.iloc[10][[('track', 'genres_all')]].values,track_df.iloc[top_k.indices[10,:]][('track','genres_all')].values))

    # get each topk matching song's metadata. results in a dataframe of shape (num_test*k, num_metadata)
    # also get the indexes of the top k results for each test song
    top_k_train_encoded = encoded_train[top_k.indices.numpy().flatten()]

    top_metadata = []
    for i in range(len(ordered_test_metadata)):
        # print(top_k.indices.shape)
        top_k_for_sample = top_k.indices[i,:]
        # these indices refer to the encoded data. use index_to_track_id to get the track ids of the top k results
        top_k_for_sample_track_ids = index_to_track_id.iloc[top_k_for_sample]["track_id"]
        # use track ids to get the metadata of the top k results
        top_k_for_sample_metadata = track_df[track_df[("general","track_id")].isin(top_k_for_sample_track_ids)]
        top_metadata.append(top_k_for_sample_metadata)
        
        # add to the top k results for each test song

    # convert list of dataframes to one dataframe
    top_metadata = pd.concat(top_metadata)
        
    utils.diagnostic_print("Shape of top k metadata for each test: {}".format(top_metadata.shape))

    # get top k genres and artists for each test song
    top_k_genres = top_metadata[[('track','genres_all')]].values
    top_k_artists = top_metadata[[('artist','name')]].values
    # exit()
    # compare genres of top suggestions with actual test genres
    top_k_genres = top_metadata["all_genres"].values
    utils.diagnostic_print("Shape of top k genres for each test: {}".format(top_k_genres.shape))

    # get the genres and artists of the test songs using their TRACK_IDS
    utils.diagnostic_print("Shape of test genres: {}".format(test_genres.shape))

    # get the number of matching genres for each test song
    # print([set(test_genres[i][0].replace('[', '').replace(" ","").replace(']','').split(",")) for i in range(len(test_genres))])
    # in test_genres, we have the genres for each test song. in top_k_genres, we have the genres for each top k result for each test song as items 1-10, 11-20, etc.
    # put all genres in each set of 10 train songs into a list, then compare that list to the genres of the test song to get the number of matching genres for each test song
    # 1. combine the genres and artists of each top k result for each test 
    
    top_k_genres_combined = []
    top_k_artists_combined = []
    top_3_genres_combined = []
    top_3_artists_combined = []
    for i in range(len(top_k_genres)):
        top_k_genres_combined.append(top_k_genres[i][0].replace('[]', '').replace(" ","").replace(']','').split(","))
        top_k_artists_combined.append(top_k_artists[i][0].replace('{', '').replace(" ","").replace('}','').split(","))
        top_3_genres_combined.append(top_k_genres[i][0].replace('[', '').replace(" ","").replace(']','').split(","))
        top_3_artists_combined.append(top_k_artists[i][0].replace('{', '').replace(" ","").replace('}','').split(","))

    # now merge each k predictions for each test song into one list
    top_k_genres_combined = [set(itertools.chain.from_iterable(top_k_genres_combined[i:i+k])) for i in range(0, len(top_k_genres_combined), k)]
    top_k_artists_combined = [set(itertools.chain.from_iterable(top_k_artists_combined[i:i+k])) for i in range(0, len(top_k_artists_combined), k)]
    top_3_genres_combined = [set(itertools.chain.from_iterable(top_3_genres_combined[i:i+3])) for i in range(0, len(top_3_genres_combined), k)]
    top_3_artists_combined = [set(itertools.chain.from_iterable(top_3_artists_combined[i:i+3])) for i in range(0, len(top_3_artists_combined), k)]

    print(top_3_artists_combined)

    utils.diagnostic_print("Shape of combined top k genres: {}".format(len(top_k_genres_combined)))

    # 2. compare the genres of each test song with the combined genres of the top k results for each test song
    # print(test_genres[0][0].replace('[', '').replace(" ","").replace(']','').split(","), top_k_genres_combined[0])
    num_matching_genres = [len(set(test_genres[i][0].replace('[', '').replace(" ","").replace(']','').split(",")).intersection(top_k_genres_combined[i])) for i in range(len(test_genres))]
    num_matching_artists = [len(set(test_artists[i][0].replace('[', '').replace(" ","").replace(']','').split(",")).intersection(top_k_artists_combined[i])) for i in range(len(test_artists))]
    num_3_matching_genres = [len(set(test_genres[i][0].replace('[', '').replace(" ","").replace(']','').split(",")).intersection(top_3_genres_combined[i])) for i in range(len(test_genres))]
    num_3_matching_artists = [len(set(test_artists[i][0].replace('[', '').replace(" ","").replace(']','').split(",")).intersection(top_3_artists_combined[i])) for i in range(len(test_artists))]

    utils.diagnostic_print("Shape of num matching genres: {}".format(len(num_matching_genres)))
    utils.diagnostic_print("Num matching genres: {}".format(num_matching_genres))

    # print percentage of times the autoencoder got 1 or more genres correct out of all test songs
    percentage_matching_genres = sum([1 for i in num_matching_genres if i > 0]) / len(num_matching_genres)
    utils.diagnostic_print("Percentage of times the autoencoder got 1 or more genres correct out of all test songs: {}".format(percentage_matching_genres))

    # for all the test instances with only one genre, print the percentage of times the autoencoder got the genre correct
    # get the number of test songs with only one genre
    num_test_songs_one_genre = sum([1 for i in test_genres if len(i[0].replace('[', '').replace(" ","").replace(']','').split(",")) == 1])
    utils.diagnostic_print("Number of test songs with only one genre: {}".format(num_test_songs_one_genre))
    # get the number of times the autoencoder got the genre correct for all test songs with only one genre
    num_correct_one_genre = sum([1 for i in range(len(num_matching_genres)) if num_matching_genres[i] == 1 and len(test_genres[i][0].replace('[', '').replace(" ","").replace(']','').split(",")) == 1])
    utils.diagnostic_print("Number of times the autoencoder got the genre correct for all test songs with only one genre: {}".format(num_correct_one_genre))
    # now get that out of all test songs with only one genre
    percentage_correct_one_genre = num_correct_one_genre / num_test_songs_one_genre
    utils.diagnostic_print("Percentage of times the autoencoder got the genre correct for all test songs with only one genre: {}".format(percentage_correct_one_genre))

    # calculate the same above for artists
    # get the percentage of times the autoencoder got 1 or more artists correct out of all test songs
    percentage_matching_artists = sum([1 for i in num_matching_artists if i > 0]) / len(num_matching_artists)
    utils.diagnostic_print("Percentage of times the autoencoder got 1 or more artists correct out of all test songs: {}".format(percentage_matching_artists))

    # get the number of test songs with only one artist
    num_test_songs_one_artist = sum([1 for i in test_artists if len(i[0].replace('[', '').replace(" ","").replace(']','').split(",")) == 1])
    utils.diagnostic_print("Number of test songs with only one artist: {}".format(num_test_songs_one_artist))
    # get the number of times the autoencoder got the artist correct for all test songs with only one artist
    num_correct_one_artist = sum([1 for i in range(len(num_matching_artists)) if num_matching_artists[i] == 1 and len(test_artists[i][0].replace('[', '').replace(" ","").replace(']','').split(",")) == 1])
    utils.diagnostic_print("Number of times the autoencoder got the artist correct for all test songs with only one artist: {}".format(num_correct_one_artist))
    # now get that out of all test songs with only one artist
    percentage_correct_one_artist = num_correct_one_artist / num_test_songs_one_artist
    utils.diagnostic_print("Percentage of times the autoencoder got the artist correct for all test songs with only one artist: {}".format(percentage_correct_one_artist))

    # want to include parents in classification. add parents of all predicted genres to the list of predicted genres
    for i in range(len(top_k_genres_combined)):
        for genre in top_k_genres_combined[i]:
            genre = genre.replace(" ","").replace("[","").replace("]","")
            if genre == "":
                continue
            if int(genre) in genres_df.keys():
                # get the last x until you hit ""
                print(genres_df[int(genre)].to_string().split(" "))
                parent_name = genres_df[int(genre)].to_string().split(" ")
                # traverse backwards grabbing the last x until you hit ""
    
                # get parent id. wherever genres_df.values.iloc[0] == parent_name
                gens = [list(genres_df.values())[i].iloc[0] for i in range(len(genres_df.values()))]
                # get index of parent name
                print(gens, parent_name)
                ix = gens.index(parent_name)
                parent_id = list(genres_df.keys())[ix]
                top_k_genres_combined[i] = top_k_genres_combined[i].union([parent_id])

    # now get number of matching genres again and percentage of correct per genre
    num_matching_genres = [len(set(test_genres[i][0].replace('[', '').replace(" ","").replace(']','').split(",")).intersection(top_k_genres_combined[i])) for i in range(len(test_genres))]
    utils.diagnostic_print("Num matching genres: {}".format(num_matching_genres))
    percentage_matching_genres = sum([1 for i in num_matching_genres if i > 0]) / len(num_matching_genres)
    utils.diagnostic_print("Percentage of times the autoencoder got 1 or more genres correct out of all test songs: {}".format(percentage_matching_genres))
    
    # get the probability of getting each genre correct for all test songs
    num_each_genre = {}
    for i in range(len(test_genres)):
        for genre in test_genres[i][0].replace('[', '').replace(" ","").replace(']','').split(","):
            if genre in num_each_genre:
                num_each_genre[genre] += 1
            else:
                num_each_genre[genre] = 1
    num_predicted_each_genre = {}
    for i in range(len(top_k_genres_combined)):
        for genre in top_k_genres_combined[i]:
            if genre in num_predicted_each_genre:
                num_predicted_each_genre[genre] += 1
            else:
                num_predicted_each_genre[genre] = 1

    utils.diagnostic_print("Num each genre: {}".format(num_each_genre))
    utils.diagnostic_print("Num predicted each genre: {}".format(num_predicted_each_genre))

    # only calculate percent correct for genres that are in the test set
    utils.diagnostic_print("Percent correct for each genre: {}".format({genre: num_predicted_each_genre[genre] / num_each_genre[genre] for genre in num_each_genre if genre in num_predicted_each_genre}))
    pc = {genre: num_predicted_each_genre[genre] / num_each_genre[genre] for genre in num_each_genre if genre in num_predicted_each_genre}
    # produce another plot
    labels = [genres_df[int(i)].iloc[0] for i in pc.keys() if i != "" and int(i) in genres_df.keys()]
    values = [val for key, val in pc.items() if key != "" and int(key) in genres_df.keys()]
    plt.bar(labels, values, color=(0.2, 0.4, 0.6, 0.6))
    plt.xticks(rotation=90)
    plt.title("Percent of songs in each genre that were predicted correctly, including parent genres")
    plt.xlabel("Genre")
    plt.ylabel("Percent of songs predicted correctly")
    plt.show()

    exit()

    # 1. Print the number of matching genres out of true genres (percent matching) for each test song in top 3 and top k
    num_true_genres = [len(test_genres[i][0].replace('[', '').replace(" ","").replace(']','').split(",")) for i in range(len(test_genres))]
    utils.diagnostic_print("!Percent matching genres for top 3: {}".format([num_3_matching_genres[i] / num_true_genres[i] for i in range(len(num_3_matching_genres))]))
    utils.diagnostic_print("!Percent matching genres for top k: {}".format([num_matching_genres[i] / num_true_genres[i] for i in range(len(num_matching_genres))]))
    # 2. Same with artists
    num_true_artists = [len(test_artists[i][0].replace('[', '').replace(" ","").replace(']','').split(",")) for i in range(len(test_artists))]
    utils.diagnostic_print("!Percent matching artists for top 3: {}".format([num_3_matching_artists[i] / num_true_artists[i] for i in range(len(num_3_matching_artists))]))
    utils.diagnostic_print("!Percent matching artists for top k: {}".format([num_matching_artists[i] / num_true_artists[i] for i in range(len(num_matching_artists))]))
    # 3. Print probability of getting each genre correct for all test songs (ex: for rock, 80% of the time it was predicted correctly)
    num_each_genre = {}
    for i in range(len(test_genres)):
        for genre in test_genres[i][0].replace('[', '').replace(" ","").replace(']','').split(","):
            if genre in num_each_genre:
                num_each_genre[genre] += 1
            else:
                num_each_genre[genre] = 1
    num_predicted_each_genre = {}
    for i in range(len(top_k_genres_combined)):
        for genre in top_k_genres_combined[i]:
            if genre in num_predicted_each_genre:
                num_predicted_each_genre[genre] += 1
            else:
                num_predicted_each_genre[genre] = 1

    utils.diagnostic_print("Num each genre: {}".format(num_each_genre))
    # bar graph of num each genre total. colorful 
    print(num_each_genre.keys(), genres_df.keys())
    labels = [genres_df[int(i)].iloc[0] for i in num_each_genre.keys() if i != "" and int(i) in genres_df.keys()]
    print(type(labels[0]))
    values = [val for key, val in num_each_genre.items() if key != "" and int(key) in genres_df.keys()]
    print(labels, values)
    plt.bar(labels, values, color=(0.2, 0.4, 0.6, 0.6))
    plt.xticks(rotation=90)
    plt.title("Number of songs in each genre")
    plt.xlabel("Genre")
    plt.ylabel("Number of songs")
    plt.show()

    utils.diagnostic_print("Num predicted each genre: {}".format(num_predicted_each_genre))


    # only calculate percent correct for genres that are in the test set
    utils.diagnostic_print("Percent correct for each genre: {}".format({genre: num_predicted_each_genre[genre] / num_each_genre[genre] for genre in num_each_genre if genre in num_predicted_each_genre}))
    pc = {genre: num_predicted_each_genre[genre] / num_each_genre[genre] for genre in num_each_genre if genre in num_predicted_each_genre}
    # produce another plot
    labels = [genres_df[int(i)].iloc[0] for i in pc.keys() if i != "" and int(i) in genres_df.keys()]
    values = [val for key, val in pc.items() if key != "" and int(key) in genres_df.keys()]
    plt.bar(labels, values, color=(0.2, 0.4, 0.6, 0.6))
    plt.xticks(rotation=90)
    plt.title("Percent of songs in each genre that were predicted correctly")
    plt.xlabel("Genre")
    plt.ylabel("Percent of songs predicted correctly")
    plt.show()

    # 4. Same with artists
    num_each_artist = {}
    for i in range(len(test_artists)):
        for artist in test_artists[i][0].replace('[', '').replace(" ","").replace(']','').split(","):
            if artist in num_each_artist:
                num_each_artist[artist] += 1
            else:
                num_each_artist[artist] = 1
    num_predicted_each_artist = {}
    for i in range(len(top_k_artists_combined)):
        for artist in top_k_artists_combined[i]:
            if artist in num_predicted_each_artist:
                num_predicted_each_artist[artist] += 1
            else:
                num_predicted_each_artist[artist] = 1
    
    utils.diagnostic_print("Num each artist: {}".format(num_each_artist))
    utils.diagnostic_print("Num predicted each artist: {}".format(num_predicted_each_artist))
    # only calculate percent correct for artists that are in the test set
    utils.diagnostic_print("Percent correct for each artist: {}".format({artist: num_predicted_each_artist[artist] / num_each_artist[artist] for artist in num_each_artist if artist in num_predicted_each_artist}))

    # do just the plot for the percentage
    pc = {artist: num_predicted_each_artist[artist] / num_each_artist[artist] for artist in num_each_artist if artist in num_predicted_each_artist}
    labels = [artist for artist in pc.keys()]
    values = [val for key, val in pc.items()]
    plt.bar(labels, values, color=(0.2, 0.4, 0.6, 0.6))
    plt.xticks(rotation=90)
    plt.title("Percent of songs by each artist that were predicted correctly")
    plt.xlabel("Artist")
    plt.ylabel("Percent of songs predicted correctly")
    plt.show()

    # 5. Print the average number of matching genres for each test song
    utils.diagnostic_print("Average number of matching genres: {}".format(sum(num_matching_genres) / len(num_matching_genres)))
    # 6. Same with artists
    utils.diagnostic_print("Average number of matching artists: {}".format(sum(num_matching_artists) / len(num_matching_artists)))



    # return the top k songs with their metadata, the number of matching genres, and the percentage of matching genres
    # NOTE: top_k = (vals, indices)
    return top_k, top_metadata, num_matching_genres, num_matching_artists, percentage_matching_genres, total_accuracy_genre_based, similarities, test_genres, test_artists, track_df, index_to_track_id


# def get_top_k_mixed_ranking(similarities, test_genres, test_artists, track_df, index_to_track_id, w_cos=0.8, w_art=0.15, w_gen=0.05,k=5):
    
#     # for each test song, if a train song shares a genre with the test song, add w_gen to its score
#     # if a train song shares an artist with the test song, add w_art to its score
#     # get train genres
#     train_indices = index_to_track_id[index_to_track_id["set_split"] == "train"][["index","track_id"]]
#     train_rows = track_df[track_df[("general","track_id")].isin(train_indices["track_id"])]
#     train_genres = train_rows[[("track","genres_all")]].values
#     train_artists = train_rows[[("artist","name")]].values

#     # quick fix: remove last 3
#     train_genres = train_genres[:-3]
#     train_artists = train_artists[:-3]

#     pbar = tqdm(total=len(test_genres) * len(train_genres))
#     for i in range(len(test_genres)):
#         sample_genres = test_genres[i][0].replace('[', '').replace(" ","").replace(']','').split(",")
#         for j in range(len(train_genres)):
#             similarities[i,j] *= w_cos
#             #add to the similarity score their intersection * w_gen
#             similarities[i,j] += len(set(sample_genres).intersection(set(train_genres[j][0].replace('[', '').replace(" ","").replace(']','').split(",")))) * w_gen
#             #add to the similarity score their intersection * w_art
#             similarities[i,j] += len(set(test_artists[i][0].replace('[', '').replace(" ","").replace(']','').split(",")).intersection(set(train_artists[j][0].replace('[', '').replace(" ","").replace(']','').split(",")))) * w_art
#             pbar.update(1)

#     # get new topk per test song
#     top_k_mixed_scores = torch.topk(similarities, k=k, dim=1)
#     utils.diagnostic_print("Shape of top k mixed results for each test: {}".format(top_k_mixed_scores.indices.shape))

#     print(top_k_mixed_scores.indices[0,:])
#     return top_k_mixed_scores



# def top_k_to_csv(top_k, track_df,index_to_track_id, k,cosine_only=False):
#     """
#     Creates a csv named f'predictions{k}CosineOnly.csv' if consine_only is True, else f'predictions{k}Mixed.csv
#     with the top k results for each test song and their metadata
#     Args:
#         top_k (tensor): tensor of shape (num_test, k) where each row is the top k results for each test song
#         top_metadata (_type_): _description_
#         num_matching_genres (_type_): _description_
#         percentage_matching_genres (_type_): _description_
#     """
#     #Decide the output csv file name
#     if cosine_only:
#         output_csv_name = f'predictions{k}CosineOnly.csv'
#     else:
#         output_csv_name = f'predictions{k}Mixed.csv'
    
#     #Filter the track_df to only show the following columns : [("general", "track_id"),("album", "title"), ("artist", "name"), ("set", "split"), ("set", "subset"), ("track", "genres_all"), ("track", "title")])
    
#     filtered_track_df = track_df[[("general", "track_id"),("album", "title"), ("artist", "name"), ("set", "split"), ("set", "subset"), ("track", "genres_all"), ("track", "title")]]
    
#     #Create a dataframe to store the results with the following columns:[("general", "track_id"),("album", "title"), ("artist", "name"), ("set", "split"), ("set", "subset"), ("track", "genres_all"), ("track", "title")]
    
#     # output_df = pd.DataFrame(columns=[("general", "track_id"),("album", "title"), ("artist", "name"), ("set", "split"), ("set", "subset"), ("track", "genres_all"), ("track", "title")])
    
#     #Get the track_ids of the test songs
#     test_track_ids = index_to_track_id["test", :, "track_id"]
    
#     #Get the actual rows of the test songs in filtered track_df.
#     test_example_rows = filtered_track_df[filtered_track_df["track_id"].isin(test_track_ids)]
    
#     #Get the actual rows of the top k results in filtered track_df. for each test example
#     top_k_rows = filtered_track_df.iloc[top_k.indices.numpy().flatten()]
#     print(f'Shape of top k rows: {top_k_rows.shape}')
    
#     #Follow the format of test example + Empty Row + k train example rows + Empty Row + Empty Row + test example + Empty Row + k train example rows + Empty Row + Empty Row + ... + test example + Empty Row + k train example rows + Empty Row + Empty Row and fill in the output.csv..
#     with open(output_csv_name, 'w',encoding="utf8") as f:
#         #Write the columns first for the csv:
#         f.write("track_id,album_title,artist_name,set_split,set_subset,track_genres_all,track_title")
        
#         for i in range(len(test_example_rows)):
#             #Write the test example row
#             f.write(test_example_rows[i])
#             #Write an empty row
#             f.write("\n")
#             #Write the k train example rows
#             for j in range(k):
#                 f.write(top_k_rows[i*k + j])
#                 f.write("\n")
#             #Write two empty rows
#             f.write("\n")
#             f.write("\n")
    
#     utils.diagnostic_print(f'Successfully wrote the results to the csv file named:{output_csv_name}')
#     return True
    
    

def get_ranking_results(encoded_train,encoded_test, track_df):
    # 2. Load in the genre data
    genres_df = utils.genre_data
    # 3. Load in the track ids from the index_to_track_id.pt file
    try:
        index_to_track_id = torch.load(os.path.join(utils.data_base_dir, "index_to_track_id.pt"))
            # "index_to_track_id.pt")
    except Exception as e:
        utils.diagnostic_print("!" + "Error loading index_to_track_id.pt")
        raise e
    # 4. Get the cosine ranking results
    top_k_cosine_only, top_metadata,  num_matching_genres, num_matching_artists, percentage_matching_genres, total_accuracy_genre_based, similarities, test_genres, test_artists, track_df, index_to_track_id= get_cosine_ranking(encoded_train, encoded_test, track_df, index_to_track_id, genres_df, k=10)
    # 5. Get the mixed ranking results
    # top_k_mixed_scores, final_scores, percentage_matching_genres_mixed, total_genre_accuracy_mixed,total_artist_accuracy_mixed = get_top_k_mixed_ranking(similarities, test_genres, test_artists, track_df, index_to_track_id, w_cos=0.8, w_art=0.15, w_gen=0.05)
    
    #6. Call a function to append the results to a csv
    # process_completed1 = top_k_to_csv(top_k_cosine_only, track_df,index_to_track_id,k=10,cosine_only=True)
    # process_completed2 = top_k_to_csv(top_k_mixed_scores, track_df,index_to_track_id,k=10,cosine_only=False)
    
    # return process_completed1
    return top_metadata

if __name__ == "__main__":
    # Use for testing with dummy data

    # Create dummy data
    num_train = 100
    num_test = 10
    encoded_train = torch.randn(num_train - num_test, 128, 32, 32)
    encoded_test = torch.randn(num_test, 128, 32, 32)
    # setting the last num_test songs in train to be the same as the test songs
    encoded_train = torch.vstack((encoded_train, encoded_test))

    utils.diagnostic_print("Shape of dummy encoded_train: " + str(encoded_train.shape))
    utils.diagnostic_print("Shape of dummy encoded_test: " + str(encoded_test.shape))

    # track_df should have id and genres,artist name. Should have num_train + num_test rows(ids) to represent each song
    # for testing, make sure the genres of the last 10 songs in train match the genres of the test songs
    # so if we test the ranking, we should see the last 10 songs in train as the top results for each test song
    track_df = pd.DataFrame(columns=["track_id", "all_genres"])
    # track_ids are in order
    track_df["track_id"] = [i for i in range(num_train + num_test)]
    # make genres alphabetical 6 letter combo for easier comparison
    alphabetical = list(itertools.product("abcdef", repeat=6))[:num_test + num_train ]
    alphabetical[-num_test:] = alphabetical[-num_test*2:-num_test]
    track_df["all_genres"] = ["".join(i) for i in alphabetical]

    # make track ids random integers between 0 and num_train + num_test
    id_gen = torch.randint(0, num_train + num_test, (num_train + num_test, 1))
    # id match is a dataframe with columns: [sub_split, index, track_id]
    id_match = pd.DataFrame(columns=["sub_split", "index", "track_id"])
    id_match["track_id"] = id_gen.squeeze()
    # for more realistic test, randomly assign the train and test songs to sub_split in the correct proportions
    sub_split_list = ["train" for i in range(num_train)] + ["test" for i in range(num_test)]
    index_list = [i for i in range(num_train)] + [i for i in range(num_test)]
    # shuffle both the same way
    shuffled = list(zip(sub_split_list, index_list))
    np.random.shuffle(shuffled)
    sub_split_list, index_list = zip(*shuffled)
    
    id_match["sub_split"] = sub_split_list
    id_match["index"] = index_list

    #1. Find all the track_ids where the sub_split is test
    test_track_ids = id_match[id_match["sub_split"] == "test"]["track_id"]
    #2. Find the genres of those track_ids
    test_genres = track_df[track_df["track_id"].isin(test_track_ids)]["all_genres"]

    #3. Find the data of those track_ids, where the index listed in id_match is the index of the data in encoded_train or encoded_test
    # for now choosing last section of training data to be the test data, but really this would be random
    encoded_train[-num_test:] = encoded_test
    # last n_test indices in encoded train correspond to the rows in id_match which have sub_split = train and equivalent indices
    # use the track_ids of those rows to set them equivalent to the test data, so that the genres match and we have some top matches
    # get rows in id_match that have indices = -num_test to -1 and are in sub_split = train
    last_rows_id_match = id_match[(id_match["index"] >= num_train-num_test) & (id_match["sub_split"] == "train")]["track_id"]
    # use track ids to set the test_genres to each track_id row
    # keep ids, new genres
    print(len(track_df.loc[last_rows_id_match, "all_genres"]), len(test_genres.values) )
    track_df.loc[last_rows_id_match, "all_genres"] = test_genres.values

    # assert that the data for the last num_test rows in encoded_train match the test data
    # assert(torch.all(encoded_train[-num_test:] == encoded_test))
    # assert that the corresponding rows in track_df have the same genres as the test data
    # assert(torch.all(track_df.loc[last_rows_id_match, "all_genres"] == test_genres.values))

    top_k, track_df, num_matching_genres, percentage_matching_genres = get_cosine_ranking(encoded_train, encoded_test, track_df, id_match, None, k=10)
    # song 
    # [("general", "track_id"),("album", "title"), ("artist", "name"), ("set", "split"), ("set", "subset"), ("track", "genres_all"), ("track", "title")]


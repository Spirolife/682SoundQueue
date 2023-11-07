# PURPOSE OF FILE: Contains the scoring/ranking system for the soundqueue project.

import itertools
import torch 
import utils
import pandas as pd
import numpy as np

def get_cosine_ranking(encoded_train, encoded_test, track_df, index_to_track_id, genres_df, k=10):
    # NOTE: genres_df is for us to visually compare the genres of the top k results with the actual test genres, so we can see how well the model is doing
    # it isnt used for the actual ranking system

    # Use torch cosine similarity: https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html

    # for each test song of shape (128, 32, 32), create a matrix of size (N, 128, 32, 32) where N is the number of songs in the training set
    # then, use cosine similarity to get a matrix of size (N, 1) where each value is the cosine similarity between the test song and the corresponding training song
    cosine_similarity_model = torch.nn.CosineSimilarity(dim=2, eps=1e-6)

    # flatten the test set
    f_encoded_test = encoded_test.flatten(start_dim=1)
    # utils.diagnostic_print("Shape of flattened encoded_test: {}".format(f_encoded_test.shape))

    # flatten the train set
    f_encoded_train = encoded_train.flatten(start_dim=1)
    # utils.diagnostic_print("Shape of flattened encoded_train: {}".format(f_encoded_train.shape))

    # get the cosine similarity between each test song and each train song by duplicating the test song N times to match the train set
    big_encoded_test = f_encoded_test.unsqueeze(0).repeat(f_encoded_train.shape[0], 1, 1)
    # utils.diagnostic_print("Shape of duplicated encoded_test: {}".format(big_encoded_test.shape))

    # duplicate train set to match the new shape of the test set
    big_encoded_train = f_encoded_train.unsqueeze(1).repeat(1, f_encoded_test.shape[0], 1)
    # utils.diagnostic_print("Shape of duplicated encoded_train: {}".format(big_encoded_train.shape))

    # get the cosine similarity between each test song and each train song
    similarities = cosine_similarity_model(big_encoded_test, big_encoded_train).T
    # utils.diagnostic_print("Shape of cosine similarities: {}".format(similarities.shape))

    # for each test item, return the top k train indices
    top_k = torch.topk(similarities, k=k, dim=1)
    utils.diagnostic_print("Shape of top k results for each test: {}".format(top_k.indices.shape))
    # print("Top 1 result for each test: {}".format(top_k.values[:,0], top_k.indices[:,0]))

    # print the genres of the actual test songs and the genres of their top match.
    # index_to_track_id has columns: [sub_split, index, track_id] so accessing test songs is index_to_track_id["test", :, "track_id"]
    # then use the track_id to get the genres of the test songs, where track_df has columns: [track_id, ..., all_genres]
    # in track_df, the track_id is NOT the index, so we have to use iloc to get the row of the track_id
    test_genres = track_df[track_df.isin(index_to_track_id["test", :, "track_id"])].iloc[:, -1].values
    print("Test genres: {}".format(test_genres))
    print("Top match genres: {}".format(track_df.iloc[top_k.indices[:,0].numpy()]["all_genres"].values))

    # get each topk matching song's metadata. results in a dataframe of shape (num_test*k, num_metadata)
    top_metadata = track_df.iloc[top_k.indices.numpy().flatten()]
    utils.diagnostic_print("Shape of top k metadata for each test: {}".format(top_metadata.shape))

    # compare genres of top suggestions with actual test genres
    top_k_genres = top_metadata["all_genres"].values
    utils.diagnostic_print("Shape of top k genres for each test: {}".format(top_k_genres.shape))

    # get the genres of the test songs using their TRACK_IDS
    # using the track_ids of the t
    utils.diagnostic_print("Shape of test genres: {}".format(test_genres.shape))

    # get the number of matching genres for each test song
    num_matching_genres = [len(set(test_genres[i].split("|")).intersection(set(top_k_genres[i].split("|")))) for i in range(len(test_genres))]
    utils.diagnostic_print("Shape of num matching genres: {}".format(len(num_matching_genres)))
    utils.diagnostic_print("Num matching genres: {}".format(num_matching_genres))

    # get the percentage of matching genres for each test song
    percentage_matching_genres = [num_matching_genres[i] / len(test_genres[i].split("|")) for i in range(len(test_genres))]
    utils.diagnostic_print("Shape of percentage matching genres: {}".format(len(percentage_matching_genres)))
    utils.diagnostic_print("Percentage matching genres: {}".format(percentage_matching_genres))

    # get total accuracy
    total_accuracy = sum(percentage_matching_genres) / len(percentage_matching_genres)
    utils.diagnostic_print("Total accuracy: {}".format(total_accuracy))


    # return the top k songs with their metadata, the number of matching genres, and the percentage of matching genres
    # NOTE: top_k = (vals, indices)
    return top_k, top_metadata, num_matching_genres, percentage_matching_genres, total_accuracy

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

    # track_df should have id and genres. Should have num_train + num_test rows(ids) to represent each song
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


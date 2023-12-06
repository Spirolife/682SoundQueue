# PURPOSE OF FILE: Contains the scoring/ranking system for the soundqueue project.

import itertools

import numpy as np
import pandas as pd
import torch
import utils


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



    #####Evaluation#####

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

    # get total accuracy genre based
    total_accuracy_genre_based = sum(percentage_matching_genres) / len(percentage_matching_genres)
    utils.diagnostic_print("Total accuracy in terms of matching genres: {}".format(total_accuracy_genre_based ))


    # return the top k songs with their metadata, the number of matching genres, and the percentage of matching genres
    # NOTE: top_k = (vals, indices)
    return top_k, top_metadata, num_matching_genres, percentage_matching_genres,total_accuracy_genre_based


def get_top_k_mixed_ranking(encoded_train,encoded_test,track_df, index_to_track_id, genres_df,w_cos=0.8, w_art=0.15, w_gen=0.05,k=5):
    """
    Provides the ranking and weighted scores of the mixed ranking system.
    where we are using the cosine similarity of the encoded data and the metadata similarity (in terms of genre and artist name matches)
    By Default k= 5
    Args:
        encoded_train (_type_): _description_
        encoded_test (_type_): _description_
        track_df (_type_): _description_
        index_to_track_id (_type_): _description_
        genres_df : DataFrame containing the genres of each song.
        w_cos (float, optional): _description_. Defaults to 0.8.
        w_art (float, optional): _description_. Defaults to 0.15.
        w_gen (float, optional): _description_. Defaults to 0.05.
        k (int, optional): _description_. Defaults to 5.

    Returns:
        Tuple of : (top_k_avg_scores, final_scores, num_matching_genres, percentage_matching_genres, total_accuracy_artist_based, total_accuracy_genre_based)
    """

    # Use torch cosine similarity: https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html

    #Input shape : (N,320)
    # then, use cosine similarity to get a matrix of size (N, 1) where each value is the cosine similarity between the test song and the corresponding training song
    cosine_similarity_model = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

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

    #get test genres using track_df
    test_genres = track_df[track_df.isin(index_to_track_id["test", :, "track_id"])].iloc[:, -1].values
    print("Test genres: {}".format(test_genres))
    
    #get the test artists using track_df
    test_artists = track_df[track_df.isin(index_to_track_id["test", :, "track_id"])].iloc[:, -1].values
    # track_df[track_df.isin(index_to_track_id["test", :, "track_id"])].iloc[:, -1].values
    print("Test artists: {}".format(test_artists))
    # compare genres of top suggestions with actual test genres
    top_k_genres = track_df["all_genres"].values
    
    ### Cosine Similarity ###
    # get the cosine similarity between each test song and each train song
    similarities = cosine_similarity_model(big_encoded_test, big_encoded_train).T
    utils.diagnostic_print("Shape of cosine similarities: {}".format(similarities.shape))
    
    ### Manually Computed Features ###
    
    ###Feature 1: Total number of matching genres###
    # get the number of matching genres for each test song
    num_matching_genres = [len(set(test_genres[i].split("|")).intersection(set(top_k_genres[i].split("|")))) for i in range(len(test_genres))]
    # get the percentage of matching genres for each test song
    percentage_matching_genres = [num_matching_genres[i] / len(test_genres[i].split("|")) for i in range(len(test_genres))]
    
    ###Feature 2: Total number of matching artists###
    ##TODO: Fix this , wrong data used! for artists
    num_matching_artists = [len(set(test_genres[i].split("|")).intersection(set(top_k_genres[i].split("|")))) for i in range(len(test_genres))]
    percentage_matching_artists = [num_matching_artists[i] / len(test_genres[i].split("|")) for i in range(len(test_genres))]

    #Get Weighted Ranking of Manual and Cosine Similarity Features
    final_scores = (w_cos * similarities + w_art * percentage_matching_artists + w_gen * percentage_matching_genres)
    
    # get total accuracy genre based
    total_accuracy_genre_based = sum(percentage_matching_genres) / len(percentage_matching_genres)
    utils.diagnostic_print("Total accuracy in terms of matching genres: {}".format(total_accuracy_genre_based ))

    #get total accuracy artist based
    total_accuracy_artist_based = sum(percentage_matching_artists) / len(percentage_matching_artists)
    utils.diagnostic_print("Total accuracy in terms of matching artists: {}".format(total_accuracy_artist_based ))
    
    #Get the maximum,minimum, and median scores on cosine similarities over all test examples.
    max_cos = torch.max(similarities)
    min_cos = torch.min(similarities)
    median_cos = torch.median(similarities)
    
    #Get the maximum,minimum, and median scores on final score over all test examples.
    max_final_score = torch.max(final_scores)
    min_final_score = torch.min(final_scores)
    median_final_score = torch.median(final_scores)
    
    #get the maximum, minimum, and median % of matching artists, and maximum, minimum and median % of matching genres.. 
    max_art = torch.max(percentage_matching_artists)
    min_art = torch.min(percentage_matching_artists)
    median_art = torch.median(percentage_matching_artists)
    
    max_gen = torch.max(percentage_matching_genres)
    min_gen = torch.min(percentage_matching_genres)
    median_gen = torch.median(percentage_matching_genres)
    
    #Print all the above information as some form of statistics...
    print("Max Cosine Similarity found over all test examples: {}".format(max_cos))
    print("Minimum Cosine Similarity found over all test examples: {}".format(min_cos))
    print("Median Cosine Similarity found over all test examples: {}".format(median_cos))
    
    print("Max Manual Similarity found over all test examples: {}".format(max_final_score))
    print("Minimum Manual Similarity found over all test examples: {}".format(min_final_score))
    print("Median Manual Similarity found over all test examples: {}".format(median_final_score))
    
    print("Max % of matching artists found over all test examples: {}".format(max_art))
    print("Minimum % of matching artists found over all test examples: {}".format(min_art))
    print("Median % of matching artists found over all test examples: {}".format(median_art))
    
    print("Max % of matching genres found over all test examples: {}".format(max_gen))
    print("Minimum % of matching genres found over all test examples: {}".format(min_gen))
    print("Median % of matching genres found over all test examples: {}".format(median_gen))
    
    avg_scores = torch.mean(final_scores, dim=1)
    top_k_avg_scores = torch.topk(avg_scores, k=k, dim=1)
    
    return top_k_avg_scores, final_scores, num_matching_genres, percentage_matching_genres, total_accuracy_artist_based, total_accuracy_genre_based


def top_k_to_csv(top_k, track_df,index_to_track_id, k,cosine_only=False):
    """
    Creates a csv named f'predictions{k}CosineOnly.csv' if consine_only is True, else f'predictions{k}Mixed.csv
    with the top k results for each test song and their metadata
    Args:
        top_k (tensor): tensor of shape (num_test, k) where each row is the top k results for each test song
        top_metadata (_type_): _description_
        num_matching_genres (_type_): _description_
        percentage_matching_genres (_type_): _description_
    """
    #Decide the output csv file name
    if cosine_only:
        output_csv_name = f'predictions{k}CosineOnly.csv'
    else:
        output_csv_name = f'predictions{k}Mixed.csv'
    
    #Filter the track_df to only show the following columns : [("general", "track_id"),("album", "title"), ("artist", "name"), ("set", "split"), ("set", "subset"), ("track", "genres_all"), ("track", "title")])
    
    filtered_track_df = track_df[("general", "track_id"),("album", "title"), ("artist", "name"), ("set", "split"), ("set", "subset"), ("track", "genres_all"), ("track", "title")]
    
    #Create a dataframe to store the results with the following columns:[("general", "track_id"),("album", "title"), ("artist", "name"), ("set", "split"), ("set", "subset"), ("track", "genres_all"), ("track", "title")]
    
    # output_df = pd.DataFrame(columns=[("general", "track_id"),("album", "title"), ("artist", "name"), ("set", "split"), ("set", "subset"), ("track", "genres_all"), ("track", "title")])
    
    #Get the track_ids of the test songs
    test_track_ids = index_to_track_id["test", :, "track_id"]
    
    #Get the actual rows of the test songs in filtered track_df.
    test_example_rows = filtered_track_df[filtered_track_df["track_id"].isin(test_track_ids)]
    
    #Get the actual rows of the top k results in filtered track_df. for each test example
    top_k_rows = filtered_track_df.iloc[top_k.indices.numpy().flatten()]
    print(f'Shape of top k rows: {top_k_rows.shape}')
    
    #Follow the format of test example + Empty Row + k train example rows + Empty Row + Empty Row + test example + Empty Row + k train example rows + Empty Row + Empty Row + ... + test example + Empty Row + k train example rows + Empty Row + Empty Row and fill in the output.csv..
    with open(output_csv_name, 'w',encoding="utf8") as f:
        #Write the columns first for the csv:
        f.write("track_id,album_title,artist_name,set_split,set_subset,track_genres_all,track_title")
        
        for i in range(len(test_example_rows)):
            #Write the test example row
            f.write(test_example_rows[i])
            #Write an empty row
            f.write("\n")
            #Write the k train example rows
            for j in range(k):
                f.write(top_k_rows[i*k + j])
                f.write("\n")
            #Write two empty rows
            f.write("\n")
            f.write("\n")
    
    utils.diagnostic_print(f'Successfully wrote the results to the csv file named:{output_csv_name}')
    return True
    
    
    

def get_ranking_results(encoded_train,encoded_test):
    # 1. Load in the track metadata
    track_df = utils.track_metadata
    # 2. Load in the genre data
    genres_df = utils.genre_data
    # 3. Load in the track ids from the index_to_track_id.pt file
    try:
        index_to_track_id = torch.load("index_to_track_id.pt")
    except Exception as e:
        utils.diagnostic_print("!" + "Error loading index_to_track_id.pt")
        raise e
    # 4. Get the cosine ranking results
    top_k_cosine_only, top_metadata, num_matching_genres, percentage_matching_genres_cosine_only = get_cosine_ranking(encoded_train, encoded_test, track_df, index_to_track_id, genres_df, k=10)
    # 5. Get the mixed ranking results
    top_k_mixed_scores, final_scores, num_matching_genres, percentage_matching_genres_mixed,total_genre_accuracy_mixed,total_artist_accuracy = get_mixed_ranking(encoded_train,encoded_test,track_df, index_to_track_id, genres_df,w_cos=0.8, w_art=0.15, w_gen=0.05)
    
    #6. Call a function to append the results to a csv
    process_completed1 = top_k_to_csv(top_k_cosine_only, track_df,index_to_track_id,k=10,cosine_only=True)
    process_completed2 = top_k_to_csv(top_k_mixed_scores, track_df,index_to_track_id,k=10,cosine_only=False)
    
    return process_completed1 and process_completed2

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


#  PURPOSE OF FILE: Preprocessing functions for the soundqueue project. Datasets are loaded and preprocessed here.

import csv
import os

import librosa
import pandas
import torch
import utils
from tqdm import tqdm

# TODO: Need a tensor to metadata link, saving track_id

def get_data(overwrite_data=False):
    BASE_DIR = os.path.join('.', 'datasets', 'fma')
    #     BASE_DIR = os.path.join('..', 'datasets', 'fma')

    if overwrite_data:
        # Split the data into train, test, val
        train_data, test_data, val_data, tracks_df, genre_data = split_data(BASE_DIR)
    else:
        # Load the data from the folders
        train_data, test_data, val_data, tracks_df, genre_data = load_data(BASE_DIR)

    return train_data, test_data, val_data, tracks_df, genre_data
    

# Converts the original FMA data into a format that can be used by the autoencoder
# Splits into train, test, val, with subfolders of audio and metadata, where audio and metadata files are named the same
def split_data(BASE_DIR):
# 0. Setup
    AUDIO_DIR = os.path.join(os.path.join(BASE_DIR, 'fma_small'))

    # Load metadata and features.
    METADATA_DIR = os.path.join(BASE_DIR, 'metadata')
    tracks_path = os.path.join(METADATA_DIR, 'tracks.csv')
    genres_path = os.path.join(METADATA_DIR, 'genres.csv')

 # 1. Load metadata for reference. Also resets the csv reader to the beginning of the file.
    utils.diagnostic_print("#" + "[Loading Metadata into New Format..]")
    
    if os.path.exists(os.path.join(BASE_DIR, 'metadata.pt')):
        tracks_df = torch.load(os.path.join(BASE_DIR, 'metadata.pt'))
    else:
        # read in the csv file using pandas
        tracks_df = pandas.read_csv(tracks_path, header=[0, 1], index_col=None, skiprows=[2], encoding='utf-8', low_memory=False)
        # Get the column indices for the data we want.
        # [track_id, album_title, artist_name, set_split, set_subset, track_genres_all, track_title]
        
        # track_id is the index, so we don't need to specify it
        utils.track_metadata = [("general", "track_id"),("album", "title"), ("artist", "name"), ("set", "split"), ("set", "subset"), ("track", "genres_all"), ("track", "title")]

        # First reduce the dataframe to only the columns we want
        tracks_df = tracks_df[utils.track_metadata]
        # And remove the rows that have NaN values
        tracks_df = tracks_df.dropna()
        # And remove the rows that have words in the track_id column
        tracks_df = tracks_df[~tracks_df[("general", "track_id")].str.contains('[a-zA-Z]')]
        # be sure to change the track_id into a 6-digit string with leading zeros
        tracks_df[("general","track_id")] = tracks_df[("general","track_id")].apply(lambda x: '{0:0>6}'.format(x))

        # Try to save the dataframe into a file
        try:
            torch.save(tracks_df, os.path.join(BASE_DIR, "metadata.pt"))
        except Exception as e:
            utils.diagnostic_print("!" + "Error saving metadata dataframe")
            raise e 

# 2. Load in all the audio files and split them into train, test, val
    utils.diagnostic_print("#" + "[Converting Audio Files into New Format..]")

    converted_train = []
    converted_test = []
    converted_validation = []

    # dataframe to link index to track_id
    index_to_track_id = pandas.DataFrame(columns=["set_split", "index", "track_id"])

    lost_to_errors = 0
    # NOTE: Not the most efficient. Ideally only need to do this once, or for each new dataset.
    for folder in os.listdir(AUDIO_DIR):
        # if it is not a folder, skip
        if not os.path.isdir(os.path.join(AUDIO_DIR, folder)):
            continue
        folder_path = os.path.join(AUDIO_DIR, folder)
        pbar2 = tqdm(total=len(os.listdir(folder_path)), desc="Converting audio files in " + folder_path)
        for file in os.listdir(folder_path):
            pbar2.update(1)
            # Convert into a tensor, autoencoder-useable format
            audio_file_name = file.split(".")[0]
            audio_file_name = '{0:0>6}'.format(audio_file_name)
            converted_audio_file, lost = convert_audio(os.path.join(folder_path, file))
            if lost:
                lost_to_errors += 1
                continue
            # Save the audio file in the correct split set by checking the tracks_df. If the matching track_id has a set_split of training, save in train folder, etc.
            try:
                tracks_df.loc[int(audio_file_name)]
                sub_split = tracks_df.loc[int(audio_file_name)][("set", "split")] if tracks_df.loc[int(audio_file_name)][("set", "split")] != "training" else "train"
                if sub_split == "train":
                    converted_train.append(converted_audio_file)
                    # add to index_to_id df
                    index_to_track_id.loc[len(index_to_track_id.index)] = [sub_split, len(converted_train), tracks_df.loc[int(audio_file_name)][("general", "track_id")]]
                elif sub_split == "test":
                    converted_test.append(converted_audio_file)
                    index_to_track_id.loc[len(index_to_track_id.index)] = [sub_split, len(converted_test), tracks_df.loc[int(audio_file_name)][("general", "track_id")]]
                elif sub_split == "validation":
                    converted_validation.append(converted_audio_file)
                    index_to_track_id.loc[len(index_to_track_id.index)] = [sub_split, len(converted_validation), tracks_df.loc[int(audio_file_name)][("general", "track_id")]]
                else:
                    utils.diagnostic_print("!" + "Error: audio file " + audio_file_name + " not found in metadata, does not have a set_split")
                    continue
            except Exception as e:
                utils.diagnostic_print("!" + "Error: audio file " + audio_file_name + " not found in metadata, skipping...")
                continue
    # try saving the converted audio files
    try:
        torch.save(converted_train, os.path.join(BASE_DIR, "train.pt"))
        torch.save(converted_test, os.path.join(BASE_DIR, "test.pt"))
        torch.save(converted_validation, os.path.join(BASE_DIR, "validation.pt"))
        torch.save(index_to_track_id, os.path.join(BASE_DIR, "index_to_track_id.pt"))
    except Exception as e:
        utils.diagnostic_print("!" + "Error saving converted audio files")
        raise e

# 3. Load in all the genres
    utils.diagnostic_print("#" + "[Loading Genres into New Format..]")

    # read in the csv file using pandas
    genres_df = pandas.read_csv(genres_path, header=[0, 1], index_col=0, skiprows=[2], encoding='utf-8')

    # Get the column indices for the data we want.
    # [genre_id, title, parent]
    utils.genre_data = ["title", "parent"]

    # First reduce the dataframe to only the columns we want
    genres_df = genres_df[utils.genre_data]

    # Now we have the indices, we can load in the data.
    num_lines = len(genres_df)
    pbar3 = tqdm(total=num_lines, desc="Loading genres")
    genre_dict = {}
    for index, row in genres_df.iterrows():
        # save all the genres into a dictionary, genre_id is index column
        genre_dict[index] = row["title"]
        pbar3.update(1)

    # Put the genre_dict into a file for loading later
    try:
        torch.save(genre_dict, os.path.join(BASE_DIR, "genre_dict.pt"))
    except Exception as e:
        utils.diagnostic_print("!" + "Error saving genre_dict")

    utils.diagnostic_print("#" + "[Finished Loading Data into New Format..]")
    return converted_train, converted_test, converted_validation, tracks_df, genre_dict

# Converts an audio file into a tensor. NOTE: This is where we can change the shape of the tensor for the autoencoder
def convert_audio(dir, sr=1000):
    lost = False
    # Load the audio file
    try:
        audio, sr = librosa.load(dir, sr=sr)
    except Exception as e:
        utils.diagnostic_print("!" + "Error loading audio file: " + dir)
        return None, lost

    # Convert to tensor
    try:
        audio_tensor = torch.from_numpy(audio)
        print(audio_tensor.shape)
    except Exception as e:
        utils.diagnostic_print("!" + "Error converting audio file to tensor: " + dir)
        return None, lost

    return audio_tensor, lost

# Loads the data from the folders pre-split
def load_data(dir):
    utils.diagnostic_print("#" + "[Loading Data from Folders..]")
    # try loading metadata df
    try:
        print(f'Load Data from dir:{dir}')
        # tracks_df = torch.load(output)
        tracks_df = torch.load(os.path.join(dir, "metadata.pt"))
    except Exception as e:
        utils.diagnostic_print("!" + "Error loading metadata dataframe")
        raise e

    # try loading converted audio files
    try:
        converted_train = torch.load(os.path.join(dir, "train.pt"))
        converted_test = torch.load(os.path.join(dir, "test.pt"))
        converted_validation = torch.load(os.path.join(dir, "validation.pt"))
    except Exception as e:
        utils.diagnostic_print("!" + "Error loading converted audio files")
        raise e
    
    try:
        genre_dict = torch.load(os.path.join(dir, "genre_dict.pt"))
    except Exception as e:
        utils.diagnostic_print("!" + "Error loading genre_dict")
        raise e

    utils.diagnostic_print("#" + "[Finished Loading Data from Folders..]")
    return converted_train, converted_test, converted_validation, tracks_df, genre_dict

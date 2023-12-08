# PURPOSE OF FILE: Start of the soundqueue project.
# 1. Trains the soundqueueing model
# 2. Runs the soundqueueing model with a supplied audio file, and produces the next song in the queue

from matplotlib import pyplot as plt
import autoencoder
import preprocessing
import rankingsystem
import torch
import utils
import os
from torch import rand
import custom_dataset

import wandb

# Process of training the model:
# 1. Load the dataset test,train,validation
# 2. Preprocess the dataset to go into the autoencoder

# 3. Train the autoencoder (use wandb)
# 4. Encode the dataset (save encoded form)

# 5. Train the ranking system (use wandb)
# 6. Save the ranking system results (if we use a model, not some process)

# 7. Use the test set to produce a ranking
# 8. Evaluate the ranking
#Initialize a new wandb run
wandb.init(project="working-sound-queue", entity="sound-queue", config={
    # "learning_rate": ,
    "architecture": "Convolutional Autoencoder",
    "dataset": "FMA",
    # "epochs": 20,
    }, tags=["autoencoder", "convolutional", "fma", "preprocessing"])


# 1 and 2, load and preprocess the dataset
train_set, test_set, validation_set, train_df, genre_df = preprocessing.get_data(True)

# song_names, user_data = preprocessing.load_user_data()
# # if wrong shape, either cut off or pad with zeros
# for i in range(len(user_data)):
#     if user_data[i].shape[1] > 60:
#         user_data[i] = user_data[i][:, :60]
#     elif user_data[i].shape[1] < 60:
#         user_data[i] = torch.cat((user_data[i], torch.zeros((128, 60-user_data[i].shape[1]))), dim=1)

#NOTE: Check how the model works with dataset inputs (each tensor) of different sizes and ensure it gets truncation!

#Could Call the function with True, but that would rerun the preprocessing which would take a lot of time:
# So dont run this : train_set, test_set, validation_set, genre_data = preprocessing.get_data(True)
utils.genre_data = genre_df
print(genre_df.keys())

# Print important information about the dataset 
utils.diagnostic_print("Train set size: " + str(len(train_set)))
utils.diagnostic_print("Test set size: " + str(len(test_set)))
utils.diagnostic_print("Validation set size: " + str(len(validation_set)))

utils.diagnostic_print("Shape of audio tensor: " + str(train_set[0][0].shape))

print(f'Cuda.is_available(p): {torch.cuda.is_available()}')

# # 3 and 4, train the autoencoder and encode the dataset
encoded_train, encoded_test = autoencoder.get_encoded_data(train_set,test_set,validation_set,wandb)
print("encoded test shape", encoded_test.shape)

# encode user data
encoder = autoencoder.get_encoder(train_set,validation_set=validation_set,wandb=wandb)
# dataset for user data 
# encoded_train, encoded_user_data = autoencoder.encode_dataset(encoder, train_set, user_data, K=255, user_data=True)
# print("here", encoded_train.shape, encoded_user_data.shape)


index_to_track_id = torch.load(os.path.join(utils.data_base_dir, "index_to_track_id.pt"))

# encoder = autoencoder.get_encoder(train_set,validation_set=validation_set,wandb=wandb)

# # 5,6,7,8 train the ranking system and evaluate the ranking
# print(encoded_test.shape)
# print(type(encoded_test))
top_metadata = rankingsystem.get_ranking_results(encoded_train, encoded_test, train_df)
# top_user_metadata = rankingsystem.get_user_cosine_ranking(encoded_train, encoded_user_data, train_df, index_to_track_id, k=10)



# for the first 10 songs in the test set, display the top 10 songs in the training set

train_indices = index_to_track_id[index_to_track_id["set_split"] == "train"][["index","track_id"]]
train_rows = train_df[train_df[("general","track_id")].isin(train_indices["track_id"])]

track_ids = top_metadata[[('general', 'track_id')]]

for i in range(5):
    sample = user_data[i]
    # each 10 of track_ids is for a different song
    track_ids_for_song = track_ids.iloc[i*10:(i+1)*10]
    # get the indices of the top 10 songs
    print(track_ids_for_song)
    indices = index_to_track_id[index_to_track_id["track_id"].isin(track_ids_for_song[("general", "track_id")])]
    indices = indices["index"]
    # get the train data from train_set at those indices
    train_preds = []
    for i in range(len(indices)):
        train_preds.append(train_set[indices.iloc[i]])

    print(train_preds)
    print(sample)

    fig, axs = plt.subplots(3, 5, figsize=(6.5, 3))

    # make sure images are avged over channels
    print(sample.shape)
    print(train_preds[0].shape)

    # top center is the test song, all else in top row empty
    for j in range(5):
        if j == 2:
            axs[0, j].imshow(sample)
            axs[0, j].set_title("Test Song")
        axs[0, j].axis('off')
            

    # second row is the top 5 songs
    for j in range(5):
        axs[1, j].imshow(train_preds[j])
        axs[1, j].axis('off')
        axs[1, j].set_title("Top " + str(j+1))
    # third row is the next 5 songs
    for j in range(5):
        axs[2, j].imshow(train_preds[j+5])
        axs[2, j].axis('off')
        axs[2, j].set_title("Next " + str(j+1))
        
    plt.show()
    
# # Print the results or save them to a file or produce needed graphs and save them
# utils.diagnostic_print("Results: " + str(results))

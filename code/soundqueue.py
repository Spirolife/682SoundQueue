# PURPOSE OF FILE: Start of the soundqueue project.
# 1. Trains the soundqueueing model
# 2. Runs the soundqueueing model with a supplied audio file, and produces the next song in the queue

import autoencoder
import preprocessing
# import rankingsystem
import torch
import utils
from torch import rand

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
wandb.init(project="sound-queue", entity="sound-queue", config={
    "learning_rate": 0.001,
    "architecture": "Convolutional Autoencoder",
    "dataset": "FMA",
    "epochs": 20,
    }, tags=["autoencoder", "convolutional", "fma", "preprocessing"])


# 1 and 2, load and preprocess the dataset
train_set, test_set, validation_set, train_df, genre_df = preprocessing.get_data()

# exit()


#NOTE: Check how the model works with dataset inputs (each tensor) of different sizes and ensure it gets truncation!

#Could Call the function with True, but that would rerun the preprocessing which would take a lot of time:
#So dont run this : train_set, test_set, validation_set, genre_data = preprocessing.get_data(True)
utils.genre_data = genre_df

# Print important information about the dataset 
utils.diagnostic_print("Train set size: " + str(len(train_set)))
utils.diagnostic_print("Test set size: " + str(len(test_set)))
utils.diagnostic_print("Validation set size: " + str(len(validation_set)))

utils.diagnostic_print("Shape of audio tensor: " + str(train_set[0][0].shape))

print(f'Cuda.is_available(): {torch.cuda.is_available()}')

# # 3 and 4, train the autoencoder and encode the dataset
encoded_train, encoded_test = autoencoder.get_encoded_data(train_set,test_set,validation_set,wandb)
# encoder = autoencoder.get_encoder(train_set,validation_set=validation_set,wandb=wandb)

# # 5,6,7,8 train the ranking system and evaluate the ranking
# results = rankingsystem.get_ranking_results(encoded_train, encoded_test, validation_set)

# # Print the results or save them to a file or produce needed graphs and save them
# utils.diagnostic_print("Results: " + str(results))

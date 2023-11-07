# PURPOSE OF FILE: Start of the soundqueue project.
# 1. Trains the soundqueueing model
# 2. Runs the soundqueueing model with a supplied audio file, and produces the next song in the queue

import utils
import preprocessing
import autoencoder
import rankingsystem

# Process of training the model:
# 1. Load the dataset test,train,validation
# 2. Preprocess the dataset to go into the autoencoder

# 3. Train the autoencoder (use wandb)
# 4. Encode the dataset (save encoded form)

# 5. Train the ranking system (use wandb)
# 6. Save the ranking system results (if we use a model, not some process)

# 7. Use the test set to produce a ranking
# 8. Evaluate the ranking


# 1 and 2, load and preprocess the dataset
train_set, test_set, validation_set, genre_data = preprocessing.get_data(True)
utils.genre_data = genre_data

# Print important information about the dataset 
utils.diagnostic_print("Train set size: " + str(len(train_set)))
utils.diagnostic_print("Test set size: " + str(len(test_set)))
utils.diagnostic_print("Validation set size: " + str(len(validation_set)))

utils.diagnostic_print("Shape of audio tensor: " + str(train_set[0][0].shape))

# # 3 and 4, train the autoencoder and encode the dataset
# encoded_train, encoded_test = autoencoder.train_autoencoder(train_set, test_set)

# # 5,6,7,8 train the ranking system and evaluate the ranking
# results = rankingsystem.get_ranking_results(encoded_train, encoded_test, validation_set)

# # Print the results or save them to a file or produce needed graphs and save them
# utils.diagnostic_print("Results: " + str(results))

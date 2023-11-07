# PURPOSE OF FILE: Autoencoder model for the soundqueue project.

import torch 
import os
import utils

# Called by main.py, trains the autoencoder and returns the encoded dataset
def get_encoded_data(train_set, test_set, validation_set):
    # Check if encoded_train, encoded_test already exist
    try: 
        encoded_train = torch.load("encoded_train.pt")
        encoded_test = torch.load("encoded_test.pt")
        utils.diagnostic_print("Encoded train and test sets found, loading...")
        return encoded_train, encoded_test
    
    except:
        utils.diagnostic_print("Encoded train and test sets not found, training autoencoder...")

        encoder = get_encoder(train_set, test_set, validation_set)

        encoded_train, validation_set, encoded_test = encode_dataset(encoder, train_set, validation_set, test_set)
        # encode the train and test set and return them
        return encoded_train, encoded_test

# Gets the optimal encoder, load or train
def get_encoder():
    # 1. iteratively change autoencoder parameters
    # 2. train autoencoder on train set with parameters
    # 3. use validation set to evaluate autoencoder
    # 4. save autoencoder or load autoencoder if the above is done
    pass

# Sends the dataset through the encoder
def encode_dataset(encoder, train_set, validation_set, test_set):
    # 1. encode the train set
    # 2. encode the test set
    # 3. save the encoded train and test set for later use
    pass


class CAE:
    def __init__(self, train_set, test_set, validation_set):
        self.train_set = train_set
        self.test_set = test_set
        self.validation_set = validation_set
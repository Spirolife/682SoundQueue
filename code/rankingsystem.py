# PURPOSE OF FILE: Contains the scoring/ranking system for the soundqueue project.

import torch 

# Called by main.py, trains the ranking system and returns the results
def get_ranking_results(encoded_train, encoded_test, validation_set):
    ranking_system = get_ranking_system(encoded_train, encoded_test, validation_set)

    get_ranking_results = get_ranking(ranking_system, encoded_train, encoded_test)


def get_ranking(encoded_train, encoded_test):
    # 1. rank the test set
    # 2. return the ranking
    pass

def get_ranking_system():
    # 1. iteratively change ranking system parameters
    # 2. train ranking system on train set with parameters
    # 3. use validation set to evaluate ranking system
    # 4. save ranking system or load ranking system if the above is done
    pass

# 
class BasicRankingSystem:
    def __init__():
        pass
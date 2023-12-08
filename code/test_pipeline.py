#TODO: Write code to import all necessary libraries, such that
#1) Write a function to split input audio files into 30 second chunks
#2) Call ConvertAudio function from preprocessing.py to convert all audio files
#3) Use Librosa to read the mp3 files.. and get tensors..
#4) Use the autoencoder to encode the tensors
#5) Use the ranking system to get the results
#6) Evaluate the results
#7) Save the results
import os
import random
from code.autoencoder import Autoencoder
from code.rankingsystem import RankingSystem

import pandas as pd
import preprocessing
import torch
# import pydub
import utils
from pydub import AudioSegment
from tqdm import tqdm

import wandb


def rename_files_in_folder(folder_path,csv_path):
    """
    Given a folder path and a csv path, renames all files in the folder to their corresponding to column Title + row number in the csv file

    Args:
        folder_path (_type_): _description_
        csv_path (_type_): _description_
    """
    #Read the csv file
    df = pd.read_csv(csv_path)
    #df might have columns: Title, Artist, Genre, etc.
    
    #If need be modify df to not consider the first row describing the columns..
    
    
    #Get the column Title
    titles = df["Title"]
    total_files = len(titles)
    num_files_not_found = 0
    #Get all files in the folder
    found_file=False
    for title in titles:
        found_file = False
        for i, file in enumerate(os.listdir(folder_path)):
            #Find the file with name in title first. 
            
            #if file name contains the title and is an mp3 file
            if title in file and file.endswith(".mp3"):
                #Rename the file to Title + row number
                os.rename(os.path.join(folder_path,file),os.path.join(folder_path,title+str(i)+".mp3"))
                found_file = True
                break
        if not found_file:
            num_files_not_found += 1
            utils.diagnostic_print("Could not find file with title: " + title)

    return num_files_not_found,total_files
    
    

def split_audio_file(audio_file_path, output_folder_path):
    """
    Splits an audio file into 30 second chunks and saves them in the output folder path
    """
    #Split mp3 file into 30 second chunks
    sound = AudioSegment.from_mp3(audio_file_path)
    
    # len() and slicing are in milliseconds
    first_30_second = sound[:30000]
    remaining_time = len(sound) - 2 * 30000
    #pick a random timestep in between 30 seconds and the -30 seconds of end of file
    random_start_time_step = random.randint(0, remaining_time-30000)
    second_30_second = sound[30000+random_start_time_step:random_start_time_step+30000 + 30000]
    last_30_second = sound[-30000:]
    # writing mp3 files is a one liner
    first_30_second_out_file_name = audio_file_path.split("/")[-1].split(".")[0] + "_chunk_1.mp3"
    second_30_second_out_file_name = audio_file_path.split("/")[-1].split(".")[0] + "_chunk_2.mp3"
    last_30_second_out_file_name = audio_file_path.split("/")[-1].split(".")[0] + "_chunk_3.mp3"
    
    first_30_second.export(os.path.join(output_folder_path,first_30_second_out_file_name), format="mp3")
    second_30_second.export(os.path.join(output_folder_path,second_30_second_out_file_name), format="mp3")
    last_30_second.export(os.path.join(output_folder_path,last_30_second_out_file_name), format="mp3")
    #Save chunks in output folder path with format: <audio_file_name>_chunk_<chunk_number>.mp3

def split_audio_files_in_folder(folder_path, output_folder_path,csv_path):
    """
    Splits all audio files in a folder into 30 second chunks and saves them in the output folder path
    """
    #Call rename files in folder to rename all files in the folder to their corresponding to column Title + row number in the csv file
    num_files_not_found,total_files = rename_files_in_folder(folder_path,csv_path)
    
    #Get all files in the folder
    for file in os.listdir(folder_path):
        #Split each file
        if file.endswith(".mp3"):
            split_audio_file(os.path.join(folder_path,file),output_folder_path)
    
    #Print the number of files not found and the total number of files
    # utils.diagnostic_print("Number of files not found: " + str(num_files_not_found))

    
def test_experiments():
    """
    Tests the experiments by running the pipeline on a small subset of the dataset
    """
    
    wandb.init(project="test_output_working_sound_queue", entity="sound-queue", config={
    # "learning_rate": ,
    "architecture": "Convolutional Autoencoder",
    "dataset": "FMA",
    # "epochs": 20,
    }, tags=["autoencoder", "convolutional", "fma", "preprocessing"])

    k_val = 5 #top k files to consider for ranking system
    base_dir_for_test_files = None #TODO: Set this to the path of the folder containing the test files
    chunks_folder_path = None #TODO: Set this to the path of the folder where you want to save the chunks
    custom_csv = None #TODO: Set this to the path of the csv file containing the titles of the test files
    expected_encoded_test_file_dir = None #encoded test file dir
    #Split all files in the folder into 30 second chunks
    #TODO: Uncomment this if already done!
    split_audio_files_in_folder(base_dir_for_test_files,chunks_folder_path,"/home/abhinav/Documents/CS7643_Group_Project/data/fma_small_info.csv")
    
    train_set,_, validation_set, train_df, genre_df = preprocessing.get_data() #pass True if rerunning code from scratch so we dont have the train and test files saved 

    #test set from examples stuff:
    test_set = preprocessing.get_test_data(base_dir_for_test_files,custom_csv)

    encoded_train, encoded_test = Autoencoder.get_encoded_data(train_set,test_set,validation_set,wandb)
    
    #Run the pipeline on the chunks
    test_experiment_hyperparameters(encoded_train,encoded_test,k_val) #To Decide which hyperparameters to use for ranking system. 
    test_exp_chunk_value(encoded_train,train_set,expected_encoded_test_file_dir,base_dir_for_test_files,use_previous_files=True,k_val=k_val) #to see which input works best..
    #TODO: in the above line change use_previous_files to False if you want to run the pipeline on the chunks without using the previous files
    
    
def test_experiment_hyperparameters(encoded_train,encoded_test,k_val=5):
    #Write code for random search over different randomly sampled uniformly distributed hyperparameters(w_cos,w_art,w_gen).
    #So randomly select  3 numbers, take the sum and divide the numbers by the sum to get 3 numbers that sum to 1. and then use those as weights for the ranking system.
    for i in tqdm(range(10)): #Run the pipeline 10 times with different hyperparameters
        w_cos = random.uniform(0,1)
        w_art = random.uniform(0,1)
        w_gen = random.uniform(0,1)
        sum_total = w_cos + w_art + w_gen
        w_cos = w_cos/sum_total
        w_art = w_art/sum_total
        w_gen = w_gen/sum_total
        
        output_csv_prefix = f"out_w_cos:{w_cos}_w_art:{w_art}_w_gen:{w_gen}"
        process_done_bool = RankingSystem.get_mixed_ranking_results(output_csv_prefix,encoded_train,encoded_test,w_cos,w_art,w_gen,k_val=k_val)
        if not process_done_bool:
            utils.raise_error(f"iteration number:{i},Could not get mixed ranking results for hyperparameters: " + str(w_cos) + " " + str(w_art) + " " + str(w_gen))
        
        
def test_exp_chunk_value(encoded_train,train_set,use_previous_files,expected_encoded_test_file_dir,base_dir_for_test_files,custom_csv,k_val=5):
    """
    #Write code to test the pipeline on different chunk data
    
    """
    folder_path_1 = None #TODO: Set this to the path of the folder containing the test files chunk 1
    folder_path_2 = None #TODO: Set this to the path of the folder containing the test files chunk 2
    folder_path_3 = None #TODO: Set this to the path of the folder containing the test files chunk 3
    
    w_cos = 0.5
    w_art = 0.25
    w_gen = 0.25 #TODO: Set these to the hyperparameters that worked best in the previous test_experiment_hyperparameters function
    if use_previous_files:
        #Get all files in the folder
        for file in expected_encoded_test_file_dir:
            if file.endswith('.pt'):
                nameof_file = file.split("/")[-1].split(".")[0]
                encoded_test = torch.load(file, map_location=torch.device('cpu'))
                utils.diagnostic_print(f'{file} found, loading...')
                output_csv_prefix = nameof_file
                process_done_bool = RankingSystem.get_mixed_ranking_results(output_csv_prefix,encoded_train,encoded_test,w_cos,w_art,w_gen,k_val=k_val)
    else:
        #Train from data stuff.
        for i, folder_path in enumerate([folder_path_1,folder_path_2,folder_path_3]):
            test_data = preprocessing.get_test_data(folder_path_1,custom_csv)
            encoded_train, encoded_test = Autoencoder.get_encoded_data(train_set,test_data,val_set=test_data,wandb=wandb)
            
if nameof_file.endswith("1"):
    output_csv_prefix = f"out_w_cos:{w_cos}_w_art:{w_art}_w_gen:{w_gen}_chunk_1"
elif nameof_file.endswith("2"):
    output_csv_prefix = f"out_w_cos:{w_cos}_w_art:{w_art}_w_gen:{w_gen}_chunk_2"
elif nameof_file.endswith("3"):
    output_csv_prefix = f"out_w_cos:{w_cos}_w_art:{w_art}_w_gen:{w_gen}_chunk_3"
else:
    utils.raise_error("File name does not end with 1,2 or 3")
process_done_bool = RankingSystem.get_mixed_ranking_results(output_csv_prefix,encoded_train,encoded_test,w_cos,w_art,w_gen,k_val=5)













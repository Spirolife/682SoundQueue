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

import pandas as pd
# import pydub
import utils
from pydub import AudioSegment


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

    
    
    

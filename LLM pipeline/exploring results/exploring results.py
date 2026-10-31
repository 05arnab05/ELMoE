# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 10:37:59 2025

@author: arnab
"""
import pickle
import os
import pandas as pd


#Saving details
Experiment="MLP Transformer 2and3 output"
saving_folder=f'C:/Users/arnab/Desktop/LLM black box model interface/Results/LLM_RAG results/{Experiment}'
pickle_file = "results.pkl"

# Construct the full path for saving the pickle file
pickle_file = os.path.join(saving_folder, pickle_file)

# Load the dictionary from the pickle file
with open(pickle_file, "rb") as f:
    loaded_results = pickle.load(f)


#%% Exploring training data
model='MLP'
ML_files_folder='C:/Users/arnab/Desktop/LLM black box model interface/files/'
training_data = pd.read_csv(ML_files_folder+'Two_variable/'+f'{model}_train_data.csv')
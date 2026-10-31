# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 14:09:11 2026

@author: arnab
"""

from pathlib import Path
import pickle
import os
from joblib import load
import numpy as np
import pandas as pd


from Evaluation import mean_sparsity_scores
from main_with_training import load_config
from Main_modules import load_model_data, Evaluation_all_answers, split_test_data


# Folder where everything was saved
cfg = load_config("config.yml")
Experiment="Final_CF"#"Final_Regression_Classification_random"
Use_case, ML_task_type= next(iter(cfg["Use_Case"].items()))
Metric='Counterfactuals'

ML_models=['Ridge', 'DT', 'RF', 'MLP','Transformer']





# Convert to Path
folder = Path(f"C:/Users/arnab/Desktop/LLM black box model interface/Results/LLM_RAG results/{Experiment}/{Use_case}")

Immutables= cfg["Immutables"][Use_case]

num_outputs, folder_name=next(iter(cfg["Output_variable_dict"].items()))
ml_folder = cfg["ML_files_folder"] / folder_name



#Get entire training data
# ---- Load results.pkl ----
with open(folder / "results.pkl", "rb") as f:
    results_redoing_metrics = pickle.load(f)
    
    
for model_name in ML_models:
    # load training/test/model artifacts
    ML_training_data, meta_data_filename, ML_test_data, model_file = (
        load_model_data(model_name, str(ml_folder) + os.sep)
    )
    values = [v for k, v in results_redoing_metrics.items()if k[3] == model_name]
    LLM_results=values[0]['LLM_raw_answers']
    Baseline_results=values[0]['LLM_raw_answers_baseline']
    test_data_XY=values[0]['test_data']
    test_data_X = test_data_XY[test_data_XY.columns.intersection(LLM_results.columns)]
    
    LLM_sparsity=mean_sparsity_scores(test_data_X, LLM_results, ML_training_data, Immutables, tolerance=0.05)
    Baseline_sparsity=mean_sparsity_scores(test_data_X, Baseline_results, ML_training_data, Immutables,tolerance=0.05)
    
    print(f'Sparsity scores for {Use_case} model {model_name} is  \n Baseline_score={Baseline_sparsity} \n LLM_scroe={LLM_sparsity}')
    

    

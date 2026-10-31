# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 08:45:44 2025

@author: arnab
"""

import pickle
import json
import pandas as pd
from pathlib import Path

# Folder where everything was saved
Experiment="Final_Accuracy_095"#"Final_Regression_Classification_random"
Use_case='Healthcare'
Output_vars= "One variable"
Metric='Accuracy'
Multivar=False
Multivar_metric='MAE'

# Convert to Path
folder = Path(f"C:/Users/arnab/Desktop/LLM black box model interface/Results/LLM_RAG results/{Experiment}/{Use_case}")

# ---- Load results.pkl ----
with open(folder / "results.pkl", "rb") as f:
    results2 = pickle.load(f)

# ---- Load RAG_test_data.csv ----
test_data2 = pd.read_csv(folder / "RAG_test_data.csv")

# ---- Load prompts.pkl ----
with open(folder / "prompts.pkl", "rb") as f:
    prompt_dict2 = pickle.load(f)

# ---- Load config_used.json ----
with open(folder / "config_used.json", "r") as f:
    loaded_cfg2 = json.load(f)

print("Loaded everything successfully.")

#comparative performance dict->df



def dicts_to_df_on_common_keys(d1, d2, col_names=("value_1", "value_2")):
    """
    Create a DataFrame using common keys of two dictionaries as index
    and their corresponding values as row entries.

    Parameters
    ----------
    d1, d2 : dict
        Input dictionaries
    col_names : tuple of str
        Column names for values from d1 and d2

    Returns
    -------
    pd.DataFrame
    """
    common_keys = d1.keys() & d2.keys()

    data = {
        col_names[0]: {k: d1[k] for k in common_keys},
        col_names[1]: {k: d2[k] for k in common_keys},
    }

    return pd.DataFrame(data)


if Use_case in ('Healthcare') and Metric=='Accuracy':
    wanted = [
    "macro_Accuracy",
    "macro_Macro_F1",
    "macro_Macro_Precision",
    "macro_Macro_Recall",
]

elif Use_case in ('Networking', 'AQ')and Metric=='Accuracy':
    wanted = [
    "macro_R2",
    "macro_RMSE",
    "macro_MAE",
]

else:
    wanted = [
    "validity_rate",
    "proximity_mean",
    "sparsity_mean",
]


dfs = {}

for model_name, second_dict in results2.items():
    if Metric=='Accuracy':
        metrics_report = second_dict["metrics_report"]['overall']
        metrics_report_bl = second_dict["metrics_report_bl"]['overall']
    else:
        metrics_report = second_dict["metrics_report"][0]
        metrics_report_bl = second_dict["metrics_report_bl"][0]
    

    df = dicts_to_df_on_common_keys(
        metrics_report_bl,
        metrics_report,
        col_names=("Baseline", "LLM-MOE")
    )
    
    # keep only the rows you want
    df = df.loc[wanted]
    dfs[model_name] = df
    

multivar_dfs={}
if Multivar==True:
    for model_name, second_dict in results2.items():
        metrics_report_mv = second_dict["metrics_report"]['per_target'][Multivar_metric]
        metrics_report_bl_mv = second_dict["metrics_report_bl"]['per_target'][Multivar_metric]
        
        multivar_df=pd.DataFrame({"baseline": metrics_report_bl_mv,"llm_moe": metrics_report_mv})
        multivar_dfs[model_name] = multivar_df
        

two_var_df=multivar_dfs
        
    
    
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:38:04 2025

@author: arnab
"""

LLM_Roles_dict = {
    "Accuracy": '''You will receive a dataframe of  model predictions from a machine learning model analyzing network conditions where each row describes a scenario. 
                 Your role is to understand the behaviour of the machine learning model which is predicting CQI value for each row. 
                 Use your own reasoning and do not use code to implement a solution.
                 ''',
    
    "Counterfactuals": '''You will receive a dataframe of network features and predicted CQI labels. 
                          Your role is to generate counterfactuals by making minimal changes to feature values to increase CQI by two levels (e.g., from 10 to 12). 
                          Use your own reasoning and do not include the CQI column.''',
    
    "Feature_importance": ''''You will receive a dataframe of network features and predicted CQI labels. 
                        Your role is to quantify the importance of each feature in determining CQI using real numbers. 
                        Each feature should have a non-zero value. 
                        Use your own reasoning and do not include the target column CQI.''',
}

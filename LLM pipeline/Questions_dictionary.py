# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:38:04 2025

@author: arnab
"""

Questions_dict = {
    "Accuracy": '''What would be the machine learning model's prediction for the following input data: {input_X_data}?''',
    
    "Counterfactuals": '''The output of the machine learning model for the following input data: {input_X_data} is {input_Y_data}, what change in input state will increase {output_change_string} according to the machine learning model?''' ,
    
    "Feature_importance": ''''You will receive a dataframe of network features and predicted CQI labels. 
                        Your role is to quantify the importance of each feature in determining CQI using real numbers. 
                        Each feature should have a non-zero value. 
                        Use your own reasoning and do not include the target column CQI.''',
}

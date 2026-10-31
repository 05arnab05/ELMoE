# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:38:04 2025

@author: arnab
"""

Questions_dict = {
    "Accuracy": '''What would be the machine learning model's prediction for the following input data: {input_X_data}?''',
    
    "Counterfactuals": '''The output of the machine learning model for the following input data: {input_X_data} is {input_Y_data}, what change in input state will change {output_change_string} according to the machine learning model?
     Try to achieve the output change with minimum input data change''' ,
    
    "Feature_importance": '''Produce a signed, per-feature attribution vector for target={output_var}, aligned to the provided feature order for input data: {input_X_data}?''',
}

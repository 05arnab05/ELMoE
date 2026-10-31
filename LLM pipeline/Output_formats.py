# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:52:51 2025

@author: arnab
"""

Output_formats_dict = {
    "Accuracy": '''Provide a python list as the answer with {output_variable_number} items. Each item in the list should be a number of floating-point or integer format. no additional text opr explanation should be provided
                     ''',
    
    "Counterfactuals": '''Provide a python list as the answer with the same number of elements as an input to the Model, without any explanation or additional text. The numbers in the list can be floating-point or integer format''',
    
    "Feature_importance": '''Return the identified feature importance values in the form of a Python dictionary of dictionaries. The outer dictionary’s keys should be the index for each sample. 
                       There should an inner dictionary for each row with keys as feature names and values as the feature importance values. 
                       Include a non-zero value for each feature and exclude the CQI_actual or CQI_predicted columns. 
                       Only return the Python dictionary of dictionaries. 
                       Do not explain your solution in any way and do not include any text such as the word Python in your response before or after the Python dictionary.''',
}
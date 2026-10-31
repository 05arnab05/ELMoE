# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 08:03:30 2025

@author: arnab
"""
from joblib import dump
import json
import pandas as pd

Model_name='Ridge'

class Ridge_metrics:
    
    def __init__(self,directory,model_file):
        self.directory=directory
        self.model=model_file
        self.model_file(self.model)
        self.metadata()
    
    def model_file(self,model_file):
        dump(self.model, self.directory+f'{Model_name}_model.joblib')
    
    def metadata(self):
        
        params = self.model.get_params()

        file= self.directory+f'{Model_name}_metadata.json'
         
        with open(file, 'w') as json_file:
            json.dump(params, json_file, indent=4)    
        
    def training_data(self,X_train,X_labels,Y_train,Y_labels,predicted_label='CQI_predicted'):
        
        #Y predicted for training data
        predicted_output_TD=self.model.predict(X_train).round(3)
        predicted_df= pd.DataFrame(predicted_output_TD,columns=[predicted_label]).reset_index(drop=True)
        
        X_train = pd.DataFrame(X_train,columns=X_labels)
        Y_train = pd.DataFrame(Y_train,columns=Y_labels)
        
        train_df = pd.concat([X_train, Y_train, predicted_df], axis=1).reset_index(drop=True)
        train_df.to_csv(self.directory+f'{Model_name}_train_data.csv',index=False)


    
    def test_data(self,X_test,X_labels,Y_test,Y_labels):
        X_test = pd.DataFrame(X_test,columns=X_labels)
        Y_test = pd.DataFrame(Y_test,columns=Y_labels)

        test_data = pd.concat([X_test, Y_test], axis=1).reset_index(drop=True)

        test_data.to_csv(self.directory+f'{Model_name}_test_data.csv',index=False) 
    

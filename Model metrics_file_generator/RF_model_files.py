# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 07:57:14 2025

@author: arnab
"""
from joblib import dump
import json
import pandas as pd
from Training_data_modification_functions import aggregate_conditions

Model_name='RF'

class RF_metrics:
    
    def __init__(self,directory,model_file):
        self.directory=directory
        self.model=model_file
        self.model_file(self.model)
        self.metadata()
    
    def model_file(self,model_file):
        self.model=model_file
        dump(self.model, self.directory+f'{Model_name}_model.joblib')
    
    def metadata(self):
        params = self.model.get_params()

        file= self.directory+f'{Model_name}_metadata.json'
         
        with open(file, 'w') as json_file:
            json.dump(params, json_file, indent=4)    
        
    def training_data(self,X_train,X_labels,Y_train,Y_labels,predicted_label='CQI_predicted',skip_secondary_features=False):
        
        #Y predicted for training data
        predicted_output_TD=self.model.predict(X_train).round(3)
        predicted_df= pd.DataFrame(predicted_output_TD,columns=[predicted_label]).reset_index(drop=True)

        #add RF_conditions column to train data
        X_train = pd.DataFrame(X_train,columns=X_labels)
        Y_train = pd.DataFrame(Y_train,columns=Y_labels)
        feature_names=X_labels
        if(skip_secondary_features==False):
            RF_conditions = X_train.apply(lambda row: aggregate_conditions(self.model, row.values.flatten().tolist(), feature_names), axis=1)
            RF_conditions.name='RF_conditions'
            #Concat all train data to form train data df
            train_data = pd.concat([X_train, Y_train, predicted_df,RF_conditions], axis=1).reset_index(drop=True)
        else:
            train_data = pd.concat([X_train, Y_train, predicted_df], axis=1).reset_index(drop=True)
            
        
        #rest Y train index

        

        train_data.to_csv(self.directory+f'{Model_name}_train_data.csv',index=False) 

    
    def test_data(self,X_test,X_labels,Y_test,Y_labels):
        X_test = pd.DataFrame(X_test,columns=X_labels)
        Y_test = pd.DataFrame(Y_test,columns=Y_labels)

        test_data = pd.concat([X_test, Y_test], axis=1).reset_index(drop=True)

        test_data.to_csv(self.directory+f'{Model_name}_test_data.csv',index=False) 
    
    

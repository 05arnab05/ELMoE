# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 07:59:19 2025

@author: arnab
"""
from joblib import dump
import json
import pandas as pd

Model_name='SVR'

class SVR_metrics:
    
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
        
    def training_data(self,X_train,X_labels,Y_train,Y_labels,batch_loss_list,epochs,batch_size,predicted_label='CQI_predicted'):
        
        #batch_loss_df=history.losses after doing model.fit
        #Y predicted for training data
        predicted_output_TD=self.model.predict(X_train).round(3)
        predicted_df= pd.DataFrame(predicted_output_TD,columns=[predicted_label]).reset_index(drop=True)
        
        
        support_vector_indices = self.model.support_
        X_train = pd.DataFrame(X_train,columns=X_labels)
        Y_train = pd.DataFrame(Y_train,columns=Y_labels)
        
        train_df = pd.concat([X_train, Y_train, predicted_df], axis=1).reset_index(drop=True)
        train_df['Is_Support_Vector'] = 0  # Initialize with 0 (not a support vector)
        train_df.loc[support_vector_indices, 'Is_Support_Vector'] = 1  # Mark support vectors with 1
        return train_df


    
    def test_data(self,X_test,X_labels,Y_test,Y_labels):
        X_test = pd.DataFrame(X_test,columns=X_labels)
        Y_test = pd.DataFrame(Y_test,columns=Y_labels)

        test_data = pd.concat([X_test, Y_test], axis=1).reset_index(drop=True)

        test_data.to_csv(self.directory+f'{Model_name}_test_data.csv',index=False) 
    
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 10:24:12 2025

@author: arnab
"""
#Everything about this 
from Main_modules import load_model_data,split_test_data,Test_data_point_extractor,Prompt_creator,LLM_answers,Evaluation
from Write_toDoc import Write_to_Doc
import pandas as pd
from RAG_classes import Baseline_RAG,KNN_RAG,VecDB_RAG,Autoencoder_RAG
import numpy as np
import matplotlib as plt
from Plotting_functions import create_dot_plot

#Variables to define which level multi-output we are using



#List of ML models,LLMs,Metrics to be explained
LLMs = ["ChatGPT"]
ML_models=['MLP'] #  'Linear','SVR','MLP'
Metrics=['Counterfactuals']
RAG_Class_variables=[Baseline_RAG,Autoencoder_RAG]
RAG_db_path=''
output_upgrade_CF=[200,2000]#CQI, UL_bitrate, DL_bitrate
Output_variable_dict={ 2: 'Two_variable'}
ML_files_folder='C:/Users/arnab/Desktop/LLM black box model interface/files/'


    
if __name__ == '__main__':
    #output variablesloop
    for number, folder in Output_variable_dict.items():
        ML_files_folder=ML_files_folder+folder+'/'
        output_variables=number
        output_upgrade=output_upgrade_CF[:number]#CQI, UL_bitrate, DL_bitrate
    
        #wether you want to write a result to a file or not
        Write_results=False
        
        
        #Create blank results_dictionary where our final result table for all_combinations is stored
        results={}
        prompt_dict={}
        
        for model_name in ML_models:
            
            #Extract model files like training data, testing data etc
            training_data, meta_data_filename, test_data, model_file = load_model_data(model_name,ML_files_folder)
            
            #RAG loop
            for RAG_Class in RAG_Class_variables:
                
                #sample n number of datapoints from test data for llm question
                test_data=test_data.sample(20)
                
                #Extract input and output variables based on number of output variables defined
                X_labels= test_data.columns[:-output_variables].tolist()
                Y_labels=test_data.columns[-output_variables:].tolist()
                
                #split Test data into Test_data_X and Test_data_Y
                test_data_X, test_data_Y = split_test_data(test_data,X_labels,Y_labels)
                
                #Create RAG DB with same dimensions as test data x
                RAG_obj=RAG_Class()
                train_subset=training_data[X_labels]
                RAG_obj.Create_DB(train_subset, training_data)
                
                #iterate through metrics to create LLM answers and evaluate
                for metric in Metrics:
                    
                    #blank dictionary to store llm answers for 1 ml model for a metric
                    LLM_answers_MLmodel_metric = {llm: [] for llm in LLMs}
                    
                    #Loop over every Test data point in the Test dataset
                    for i in range(len(test_data)):
                        #Extract Test data rows
                        test_data_X_row=test_data_X.iloc[i]
                        test_data_Y_row=test_data_Y.iloc[i]
                        
                        #Convert them to string test datapoints to be used in LLM question
                        Test_datapoint_X_string,Test_datapoint_Y_string=Test_data_point_extractor(test_data_X_row,test_data_Y_row)
                        
                        #Parameters for generating prompt for LLM
                        prompt_creator_arguments= {
                            "metric": metric,
                            "ML_model": model_file,
                            "meta_data_filename": meta_data_filename,
                            "training_data": training_data,
                            "output_variable_number": output_variables,
                            "output_upgrade_CF":output_upgrade,
                            "RAG_object":RAG_obj
                        }
                        
                        #Everything about this
                        #Create prompt for LLM, based on passed parameters above
                        role,content=Prompt_creator(prompt_creator_arguments, Test_datapoint_X_string, Test_datapoint_Y_string)
                        
                        #Get answers from each LLM for a particular prompt
                        answer_one_datapoint=LLM_answers(LLMs,role,content)
                        
                        #name of RAG in string for labelling results
                        RAG_name=RAG_obj.__class__.__name__
                        prompt_dict[f'{model_name}_{metric}_{RAG_name}_{output_variables} variables']=role+content

                        #Append to dictionary to have all answers for test dataset together
                        for key, value in answer_one_datapoint.items():
                            if key in LLM_answers_MLmodel_metric:
                                LLM_answers_MLmodel_metric[key].append(value)
            
                    
                    #parameters needed for evaluation function
                    Evaluation_function_arguments= {
                        "metric": metric,
                        "LLM_predictions": LLM_answers_MLmodel_metric,
                        "test_data_X": test_data_X,
                        "test_data_Y": test_data_Y,
                        "output_variable_number": output_variables,
                        "model": model_file,
                        "model_name": model_name,
                        "X_labels":X_labels,
                        "Y_labels":Y_labels
                    
                    }
                    
                    #Evaluating metric results
                    for LLM_model in LLMs:
                        #Results for 1 LLM model
                        l1_norm,l2_norm,LLM_predictions,ML_predictions=Evaluation(LLM_model,Evaluation_function_arguments,increment=output_upgrade_CF)
                        #Structure for appending in results dictionary
                        new_result = {
                            (metric, model_name, output_variables, RAG_name, LLM_model): {
                                "l1_norm": l1_norm,
                                "l2_norm": l2_norm,
                                "ground_truth": ML_predictions,
                                "llm_results": LLM_predictions,
                                "test_data_y":test_data_Y
                            }
                            }
                        results.update(new_result)
    
                    
                    
                
                #write results of model metric to Word document
                if(Write_results==True):
                    pass
                    #Write_to_Doc(model_name, metric, role+content, Model_results)  

# Plotting comparative results of ML model and LLM for each combination of variables                
    for key, value in results.items():
        metric, model, num_outputs, retrieval, llm = key  # Unpack key values
        
        #Extracting ground_truth and LLM_results for a particular combo
        ground_truth = np.array(value["ground_truth"])  # Convert lists to NumPy arrays
        llm_results = np.array(value["llm_results"])  # Shape: (num_outputs, num_samples)

        #looping over each output variable
        for i in range(num_outputs):  # Loop over each output variable
            y1=[row[i] for row in llm_results]
            y2=[row[i] for row in ground_truth]
            # Create title of dot plot
            title = f"{metric} of {model} model with {num_outputs} output variables using {retrieval} and {llm} for {Y_labels[i]}"

            create_dot_plot(y1, f'{llm} _Predictions',y2, f'{model}_model predictions',title=title)

               

         
        
         

    


       
        
    
         
    

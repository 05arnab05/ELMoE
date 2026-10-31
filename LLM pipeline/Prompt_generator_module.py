# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:22:52 2025

@author: arnab
"""
import Dictionaries.LLM_role_dictionary as LLM_role_dictionary
import Dictionaries.Feature_and_model_description_dictionaries as Feature_and_model_description_dictionaries
import Dictionaries.Tasks as Tasks
import Dictionaries.Output_formats as Output_formats
import Dictionaries.Questions_dictionary as Questions_dictionary
import Dictionaries.Constraints_dictionary as Constraints_dictionary
import Dictionaries.JG_dictionary as JG_dictionary
import re
import numpy as np
import pandas as pd

from Trained_ML_model_functions import Load_ModelFile, Load_Scalers

from Cat_variable_functions import ( 
                                    transform_cat_encoder, prepare_NN_embedding_inputs, 
                                    postprocess_class_predictions, 
                                    Ridge_prepare_inputs_from_df,
                                    encode_query_classification)


class Prompt_generator:
    
    def __init__(self, args):
        
        self.LLM_outputs=args['LLM_outputs']
        self.ML_task_type=args['ML_task_type'] 
        self.Use_case=args['Use_case']
        self.Dice_grid=args['Dice_grid']
        self.Inputs=args['Inputs']
        self.Outputs=args['Outputs']
        self.target_cols=args['Targets']
        self.Immutables=args["Immutables"]
        self.metric=args['metric']
        self.model_name=args["ML_model_name"]
        self.model_file_path=args["ML_model_file_path"]
        self.training_data=args["training_data"]
        self.X_encoders=args["X_cat_encoder"]
        self.Y_encoders=args["Y_cat_encoder"]
        self.loaded_model=None
        self.scaler_X=None
        self.scaler_Y=None

   
    def LLM_role(self,skip=False):
        LLM_roles_dict=LLM_role_dictionary.LLM_Roles_dict[self.Use_case]
        LLM_role_prompt= LLM_roles_dict[self.metric]
        return LLM_role_prompt
    
    #Modified 
    def Dataset_info(self, skip=False):
        """
        Generates a detailed description of the dataset and model for LLM explanation purposes.
        
        Parameters
        ----------
        ML_model_name : str
            Identifier for the ML model (e.g., "DT", "RF", "MLP")
        skip : bool, optional
            Placeholder for future use; currently unused.

        Returns
        -------
        str
            A string containing model description, primary feature explanations, and secondary features.
        """
        #importing dictionaries
        Model_description_dict=Feature_and_model_description_dictionaries.Model_description_dict
        Primary_features_dict=Feature_and_model_description_dictionaries.Primary_features_dict[self.Use_case]
        Secondary_features_dict=Feature_and_model_description_dictionaries.Secondary_features_dict

        # Fetch model description
        model_description = Model_description_dict.get(self.model_name, "No description available for this model.")

        # Assemble primary features from all groups
        primary_features_description = "\n\n--- Primary Features ---\n"
        for group_name, group_description in Primary_features_dict.items():
            primary_features_description += f"\n{group_name}:\n{group_description.strip()}\n"

        # Fetch secondary features
        secondary_features_description = "\n\n--- Secondary Features ---\n"
        secondary_features_description += Secondary_features_dict.get(self.model_name, "No secondary features defined for this model.")

        # Combine all sections into a final metadata string
        Metadata = f"--- Model Description: {self.model_name} ---\n{model_description.strip()}\n"
        Metadata += primary_features_description
        Metadata += secondary_features_description

        return Metadata
    

        
    #New one
    def TD_sample_data_Clusterer(self, test_datapoint_X, test_datapoint_Y, output_upgrade_CF, output_var_num, Cluster_class_obj, skip=False):
        '''
        Input: Question (training data is already fed in create_db step in main.py)
        Output: String of most similar training data occurences
        
        RAG module to extract training data:
        in case no RAG object is passed then sample 50 points else Retreive data based RAG object provided'''
        
        Training_data_sample = "\n\n--- Sample of machine learning model training data ---\n"
        
        TD_string=Cluster_class_obj.Retreive_Cluster_expert_answers(test_datapoint_X, test_datapoint_Y, output_upgrade_CF)
        
        Training_data_sample += TD_string          
        return Training_data_sample
       
    
         
    
        
    #Modified mainly counterfactuals
    def Question(self, Test_datapoint_X_string, Test_datapoint_Y_string, output_upgrade_CF, skip=False):
        
         
        Question = "\n\n----------- Question ---------------\n"
        Questions_dict=Questions_dictionary.Questions_dict
        
        
        if(self.metric=='Counterfactuals'):
            
            if self.ML_task_type=='regression':
                prediction_values=self._get_model_predictions_regression(Test_datapoint_X_string)
            elif self.ML_task_type=='classification':
                prediction_values=self._get_model_predictions_classification(Test_datapoint_X_string)
               
            pred_arr = np.atleast_1d(prediction_values)
            
            #For prompt creation
            Output_column_names_string = ", ".join(self.Outputs)
            columns = Output_column_names_string.split(", ")
          
            if len(columns) == len(pred_arr):
                if self.ML_task_type=='regression':
                    formatted_str = ", ".join([f"{col}: {val:.2f}" for col, val in zip(columns, pred_arr)])
                elif self.ML_task_type=='classification':
                    formatted_str = ", ".join([f"{col}: {val}" for col, val in zip(columns, pred_arr)])

    
            Test_datapoint_Y_string=formatted_str
            
            #Creating part of prompt for explaining per variable change desired
            if self.ML_task_type=='regression':
                output_change_phrases = [f"{var} by +{val}" for var, val in zip(self.Outputs, output_upgrade_CF)]
            elif self.ML_task_type=='classification':
                output_change_phrases = [f"{var} to {val}" for var, val in zip(self.Outputs, output_upgrade_CF)]

            
            # Join these phrases with commas
            output_change_string = ", ".join(output_change_phrases)
            
            #Immutables string
            if len(self.Immutables)!=0:
                immutables_text = ("# The following variables are immutable and must not be mutated:\n"
                                   "# " + ", ".join(self.Immutables) + "\n")
            else:
                immutables_text=""
            

            #Creating final compiled Question prompt
            Question_prompt = Questions_dict["Counterfactuals"].format(input_X_data=Test_datapoint_X_string, input_Y_data=Test_datapoint_Y_string, output_change_string=output_change_string) + '\n\n'+immutables_text+ '\n\n'
        
        elif (self.metric=='Accuracy'):
            Output_column_names_string = ", ".join(self.Outputs)
            Question_prompt= Questions_dict[self.metric].format(input_X_data=Test_datapoint_X_string,
                                                                input_Y_data=Test_datapoint_Y_string,
                                                                output_var=Output_column_names_string,
                                                                output_upgrade_CF=output_upgrade_CF)+'\n\n'
        
        elif (self.metric=='Feature_importance'):
            Output_column_names_string = ", ".join(self.Outputs)
            Question_prompt= Questions_dict[self.metric].format(input_X_data=Test_datapoint_X_string,
                                                                input_Y_data=Test_datapoint_Y_string,
                                                                output_var=Output_column_names_string,
                                                                output_upgrade_CF=output_upgrade_CF)+'\n\n'
            
        Question=Question+ Question_prompt
        return Question
    
    def Tasks_string(self, output_variable_number,skip=False)->str:
        
        if self.metric=='Accuracy':
            Target_cols=", ".join(self.target_cols)
            Input_column_names_string = ", ".join(self.Inputs)
            num_input_features=len(self.Inputs)
            Tasks_dict=Tasks.Tasks_dict
            
            Task_string= Tasks_dict[self.LLM_outputs][self.metric].format(output_variable_number=output_variable_number,
                                                                         target_list=Target_cols,
                                                                         input_features=Input_column_names_string, 
                                                                         input_variable_number=num_input_features)+'\n\n'
        
        elif self.metric=='Counterfactuals':
            Target_cols=", ".join(self.target_cols)
            Input_column_names_string = ", ".join(self.Inputs)
            Immutables_string = ", ".join(self.Immutables)
            Non_immutable_cols=[x for x in self.Inputs if x not in self.Immutables]
            Non_immutable_col_string=", ".join(Non_immutable_cols)
            num_input_features=len(self.Inputs)
            Tasks_dict=Tasks.Tasks_dict
            
            Task_string= Tasks_dict[self.LLM_outputs][self.metric].format(output_variable_number=output_variable_number,
                                                                         target_list=Target_cols,
                                                                         input_features=Input_column_names_string, 
                                                                         immutable_features=Immutables_string,
                                                                         non_immutable_features=Non_immutable_col_string,
                                                                         input_variable_number=num_input_features)+'\n\n'
        
        return Task_string
    
    def JG_string(self,skip=False): #Judgement guidance
        
        JG_dict=JG_dictionary.JG_dict
        JG_string= JG_dict[self.metric]
        return JG_string

    def Constraints_string(self,skip=False):
        Input_column_names= ", ".join(self.Inputs)
        Input_column_number=len(self.Inputs)
        Immutable_cols_string=", ".join(self.Immutables)
        Target_column_names = ", ".join([f'"{t}": ...' for t in self.target_cols])
        Target_column_number=len(self.target_cols)
        Constraints_dict=Constraints_dictionary.Constraints_dict
        Constraints_prompt_string= Constraints_dict[self.LLM_outputs][self.metric].format(Input_column_names=Input_column_names, Input_column_number=Input_column_number, Target_column_names=Target_column_names, Target_column_number=Target_column_number, immutable_features=Immutable_cols_string)
        return Constraints_prompt_string
    
    def Output_format(self, output_variable_number, skip: bool = False) -> str:
        """
        Builds the 'Format of answer' block for prompts.
        - 'prediction' contains one number per target.
        - 'feature_importances' is now a dict keyed by target, each with a full
          feature->percentage mapping that must sum to 100 for that target.
        - 'row_weights' is included to let the LLM report neighbor-row weights.
        """
        Output_format_header = "\n\n----------- Format of answer ---------------\n"

        Target_cols = list(self.target_cols)              # e.g. ["DL_bitrate_predicted","RSRP_predicted"]
        Inputs = list(self.Inputs)                        # e.g. ["RSRQ","RSSI","Longitude","Latitude","CQI","Speed","UL_bitrate","SNR"]

        # prediction block:  "DL_bitrate_predicted": ..., "RSRP_predicted": ...
        outputs_str = ", ".join([f'"{t}": ...' for t in Target_cols])
        cf_inputs_str = ",\n    ".join([f'"{f}": <number>' for f in Inputs])

        # per-target feature importance block:
        #   "DL_bitrate_predicted": { "RSRQ": {}%, "RSSI": {}%, ... },
        #   "RSRP_predicted":       { "RSRQ": {}%, "RSSI": {}%, ... }
        feature_pairs_str = ", ".join([f'"{f}": {{}}%' for f in Inputs])
        fi_per_target_str = ",\n    ".join([f'"{t}": {{ {feature_pairs_str} }}' for t in Target_cols])
        
        # -------- per-target MAE entries (keys = base target names) ----------
        # Convert predicted column names to base names by stripping common suffixes/prefixes.
        def base_name(col: str) -> str:
            # strip common prefixes
            col = re.sub(r'^(Y_pred_|y_pred_|pred_)', '', col)
            # strip common suffixes
            col = re.sub(r'(_predicted|_ground_truth)$', '', col)
            return col
    
        base_targets = [base_name(t) for t in Target_cols]
        per_target_mae_entries = ",\n                          ".join([f'"{bt}": <number>' for bt in base_targets])

        
        # choose template from dict by metric
        Output_format_dict = Output_formats.Output_formats_dict

        Output_format_prompt = Output_format_dict[self.LLM_outputs][self.metric].format(
            cf_inputs_str=cf_inputs_str,
            output_variable_number=output_variable_number,
            outputs_str=outputs_str,
            fi_per_target=fi_per_target_str,
            per_target_mae_entries=per_target_mae_entries,
        ) + "\n\n"

        return Output_format_header + Output_format_prompt
    
    def _get_model_predictions_regression(self, model_input_string):
        
        #Formatting model input to be put into the model
        model_input = [float(num) for num in re.findall(r":\s*(-?\d+(?:\.\d+)?)", model_input_string)]
        model_input=  np.array(model_input).reshape(1, -1)
        # 0) Standardize incoming array to (N,F) float32 once
        X2d = _as_2d_float32(model_input)

        # 1) Load (cached) model & scalers
        if getattr(self, "loaded_model", None) is None:
            self.loaded_model = Load_ModelFile(self.model_file_path)
        model = self.loaded_model

        scaler_X = scaler_Y = None
        if self.model_name in ("MLP", "CNN", "Transformer",'Ridge'):
            # cache scalers too if you want (not shown)
            scaler_X, scaler_Y = Load_Scalers(self.model_file_path)

        # 2) Route by model and enforce the ONE correct shape per model
        if self.model_name in ('RF', 'DT'):
            preds = model.predict(X2d)                 # (N,O)
            return np.asarray(preds, dtype=float).reshape(preds.shape[0], -1).squeeze()

        elif self.model_name in ('MLP', 'Transformer', 'Ridge'):
            X_scaled = scaler_X.transform(X2d).astype(np.float32) # (N,F) float32
            preds = model.predict(X_scaled)
            return scaler_Y.inverse_transform(preds).reshape(preds.shape[0], -1).squeeze()

        else:
            # Fallback: assume (N,F) float32 is what the model wants
            preds = model.predict(X2d, verbose=0)
            return np.asarray(preds, dtype=float).reshape(preds.shape[0], -1).squeeze()
        
    def _get_model_predictions_classification(self, model_input_string, feature_names=None):
        """
        Classification version of _get_model_predictions_regression.

        - Same input idea: model_input can be 1D/2D array-like or a DataFrame.
        - feature_names only needed if model_input is not a DataFrame.
        - Uses:
            - transform_cat_encoder
            - prepare_NN_embedding_inputs (for MLP/Transformer)
            - postprocess_class_predictions
        - Returns: numpy array, squeezed like the regression helper.
        """
        
        X_df = encode_query_classification(model_input_string)
        
        # 1) Load (cached) model
        if getattr(self, "loaded_model", None) is None:
            self.loaded_model = Load_ModelFile(self.model_file_path)
        model = self.loaded_model

        # 2) Encode categorical features
        df_encoded = transform_cat_encoder(X_df, self.X_encoders)

        # 3) Branch by model_name
        if self.model_name in ("MLP", "Transformer"):
            # load scalers
            scaler_X, scaler_Y = Load_Scalers(self.model_file_path)

            # prepare embedding inputs
            nn_inputs = prepare_NN_embedding_inputs(
                df_encoded,
                cat_encoders=self.X_encoders,
                scaler_X=scaler_X,
                feature_names=feature_names,
            )

            raw_preds = model.predict(nn_inputs, verbose=0)

            preds_list = postprocess_class_predictions(
                raw_preds,
                self.target_cols,
                self.Y_encoders,
                mode="Neural_Net",
            )

        elif self.model_name == "Ridge":
            scaler_X, scaler_Y = Load_Scalers(self.model_file_path)
            df_prep = Ridge_prepare_inputs_from_df(X_df, scaler_X, self.X_encoders)
            raw_preds = model.predict(df_prep)
            preds_list = postprocess_class_predictions(
                raw_preds,
                self.target_cols,
                self.Y_encoders,
            )

        elif self.model_name in ("RF", "DT"):
            df_prep = df_encoded
            raw_preds = model.predict(df_prep)
            preds_list = postprocess_class_predictions(
                raw_preds,
                self.target_cols,
                self.Y_encoders,
            )

        else:
            # generic classifier fallback
            df_prep = df_encoded
            raw_preds = model.predict(df_prep)
            preds_list = postprocess_class_predictions(
                raw_preds,
                self.target_cols,
                self.Y_encoders,
            )

        # 4) Match regression helper's output shape style
        preds_arr = np.asarray(preds_list, dtype=object).reshape(len(preds_list), -1).squeeze()

        return preds_arr
    
            
    

    def _output_format_old(self,output_variable_number,skip=False):
        Target_cols=self.target_cols
        Input_column_names_string = ", ".join(self.Inputs)
        num_input_features=len(self.Inputs)
        Output_format_dict=Output_formats.Output_formats_dict
        
        Output_format_prompt= Output_format_dict[self.metric].format(output_variable_number=output_variable_number,
                                                                     target_list=Target_cols,
                                                                     input_features=Input_column_names_string, 
                                                                     input_variable_number=num_input_features)+'\n\n'
        return Output_format_prompt



#-------------HELPERS--------------
def _as_2d_float32(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)          # (1, F) for single sample
    elif x.ndim != 2:
        raise ValueError("Expected 2D (N,F) input before shaping.")
    return x

def _as_2d_dataframe(x, feature_names=None):
    """
    Normalize input to a 2D pandas DataFrame:

    - If x is already a DataFrame: return it (optionally reordered to feature_names).
    - If x is 1D: reshape to (1, F).
    - If x is 2D: keep shape.
    - If x is not a DataFrame, feature_names must be provided so we can name columns.

    This keeps raw values (including strings) intact for categorical encoders.
    """
    if isinstance(x, pd.DataFrame):
        if feature_names is None:
            return x
        # Reorder / subset to match expected feature order
        missing = [c for c in feature_names if c not in x.columns]
        if missing:
            raise ValueError(f"The following feature_names are not in DataFrame columns: {missing}")
        return x[feature_names]

    # Otherwise, treat as array-like
    arr = np.asarray(x, dtype=object)  # keep strings, categories, etc.

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim != 2:
        raise ValueError("Expected 1D or 2D input for classification model.")

    N, F = arr.shape

    if feature_names is None:
        raise ValueError("feature_names must be provided when model_input is not a DataFrame.")

    if len(feature_names) != F:
        raise ValueError(
            f"feature_names length = {len(feature_names)}, but model_input has {F} columns"
        )

    return pd.DataFrame(arr, columns=feature_names)


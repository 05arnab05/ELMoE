# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:52:51 2025

@author: arnab
"""

Output_formats_dict = {
        "One_output":{
            
                       "Accuracy": '''Output format (JSON only):
                                        {{
                                          "prediction": {{
                                            {outputs_str}
                                          }},
                                          
                                        }}
                                        ''',
                        
                        "Counterfactuals": '''Output format (JSON only):
                                                {{
                                                    "counterfactual_input": {{
                                                        {cf_inputs_str}
                                                        }},
                                                    }}''',
                        
                        "Feature_importance": '''Provide a Python list with {output_variable_number} items.  
                                                Each item must be a Python list of {input_variable_number} signed feature-importance values.  
                                                    
                                                    Constraints:
                                                    - Outer length = {output_variable_number} (targets in fixed order: {target_list}).  
                                                    - Each inner list length = {input_variable_number} (features in fixed order: {input_features}).  
                                                    - Numbers only (float or int).  
                                                    - Each inner list must be L1-normalized: sum of absolute values = 1.0 (±0.5%).  
                                                    - Signs: positive → increases that target, negative → decreases.  
                                                    - Output ONLY the raw Python list-of-lists, no text.''',
                    },
        
        "Two_output":{
            
                       "Accuracy": '''Output format (JSON only):
                                        {{
                                          "prediction": {{
                                            {outputs_str}
                                          }},
                                          "feature_importances": {{
                                            {fi_per_target}
                                          }},
                                        }}
                                        ''',
                        
                        "Counterfactuals": '''Output format (JSON only):
                                                {{
                                                    "counterfactual_input": {{
                                                        {cf_inputs_str}
                                                        }},
                                                    "feature_importances": {{
                                                        {fi_per_target}
                                                        }}
                                                    }}''',
                        
                        "Feature_importance": '''Provide a Python list with {output_variable_number} items.  
                                                Each item must be a Python list of {input_variable_number} signed feature-importance values.  
                                                    
                                                    Constraints:
                                                    - Outer length = {output_variable_number} (targets in fixed order: {target_list}).  
                                                    - Each inner list length = {input_variable_number} (features in fixed order: {input_features}).  
                                                    - Numbers only (float or int).  
                                                    - Each inner list must be L1-normalized: sum of absolute values = 1.0 (±0.5%).  
                                                    - Signs: positive → increases that target, negative → decreases.  
                                                    - Output ONLY the raw Python list-of-lists, no text.''',
                    },
        
        
        "More_output":{
                            "Accuracy": '''Output format (JSON only):
                                             {{
                                               "prediction": {{
                                                 {outputs_str}
                                               }},
                                               "feature_importances": {{
                                                 {fi_per_target}
                                               }},
                                               "local_error": {{
                                                 "MAE": {{
                                                   {per_target_mae_entries}
                                                 }}
                                               }}
                                             }}
                                          ''',
                         
                            "Counterfactuals": '''Output format (JSON only):
                                                     {{
                                                       "counterfactual_input": {{
                                                         {cf_inputs_str}
                                                       }},
                                                       "feature_importances": {{
                                                         {fi_per_target}
                                                       }},
                                                       "local_error": {{
                                                         "MAE": {{
                                                           {per_target_mae_entries}
                                                         }}
                                                       }}
                                                     }}''',
                         
                            "Feature_importance": '''Provide a Python list with {output_variable_number} items.  
                                                     Each item must be a Python list of {input_variable_number} signed feature-importance values.  
                                                         
                                                         Constraints:
                                                         - Outer length = {output_variable_number} (targets in fixed order: {target_list}).  
                                                         - Each inner list length = {input_variable_number} (features in fixed order: {input_features}).  
                                                         - Numbers only (float or int).  
                                                         - Each inner list must be L1-normalized: sum of absolute values = 1.0 (±0.5%).  
                                                         - Signs: positive → increases that target, negative → decreases.  
                                                         - Output ONLY the raw Python list-of-lists, no text.''',
                         }
                                           
         
            
            }
    


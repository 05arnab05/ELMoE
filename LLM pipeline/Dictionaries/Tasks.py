# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:52:51 2025

@author: arnab
"""

Tasks_dict = {
    'One_output':{
                    "Accuracy": '''Task:
                                    - Step 1: Use ONLY the retrieved data below to estimate the model’s prediction for the query row.
                                    - Return JSON ONLY matching the schema below. If sums differ due to rounding, renormalize to exactly 100.00.
                                     ''',
                    
                    "Counterfactuals": '''Task:
                                        - Step 1: Use ONLY the retrieved data below to construct a SINGLE counterfactual
                                        input vector for the query row, with minimum input changes to achieve required output change.
                                        • For IMMUTABLE features
                                          {immutable_features}:
                                            - DO NOT change them.
                                            - Copy the original query row value exactly into "counterfactual_input".
                                        • For MUTABLE features
                                          {non_immutable_features}:
                                            - You may change these values to achieve the intended change
                                              in LOS_CLASS (from Medium to Short).
          
                                      - Return JSON ONLY matching the schema below. If sums differ due to rounding, renormalize to exactly 100.00.
                    ''',
                    
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
    
    'Two_output':{
                    "Accuracy": '''Task:
                                    - Step 1: Use ONLY the retrieved data below to estimate the model’s prediction for the query row.
                                      
                                    - Step 2: For EACH target in {target_list}, assign a separate set of feature importances
                                              across the input features {input_features}.
                                    - Step 3: Each target’s feature importances must be non-negative percentages that SUM TO 100 (per target).
                                    - Return JSON ONLY matching the schema below. If sums differ due to rounding, renormalize to exactly 100.00.
                                     ''',
                    
                    "Counterfactuals": '''Task:
                                        - - Step 1: Use ONLY the retrieved data below to construct a SINGLE counterfactual
                                        input vector for the query row, with minimum input changes to achieve required output change.
                                        • For IMMUTABLE features
                                          {immutable_features}:
                                            - DO NOT change them.
                                            - Copy the original query row value exactly into "counterfactual_input".
                                        • For MUTABLE features
                                          {non_immutable_features}:
                                            - You may change these values to achieve the intended change
                                              in LOS_CLASS (from Medium to Short).
                                      - Step 2: For EACH target in {target_list}, assign a separate set of feature importances
                                          across the input features {input_features}.
                                      - Step 3: Each target’s feature importances must be non-negative percentages that SUM TO 100 (per target).
                                      - Return JSON ONLY matching the schema below. If sums differ due to rounding, renormalize to exactly 100.00.
                    ''',
                    
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
    
    'More_output':{
                        "Accuracy": '''Task:
                                        - Step 1: Use ONLY the retrieved data below to estimate the model’s prediction for the query row.
                                        - Step 2: For EACH target in {target_list}, assign a separate set of feature importances
                                                  across the input features {input_features}.
                                        - Step 3: Each target’s feature importances must be non-negative percentages that SUM TO 100 (per target).
                                        - Step 4 (Local error, per target): For EACH target name T in {target_list},
                                                find the columns "T_ground_truth" and "T_predicted" in the retrieved rows and compute
                                                MAE[T] = mean( abs( row["T_ground_truth"] − row["T_predicted"] ) ) over all usable rows
                                                (exclude missing/invalid; exclude the baseline row if instructed).
                                        - Return JSON ONLY matching the schema below. If sums differ due to rounding, renormalize to exactly 100.00.
                                     ''',
                    
                        "Counterfactuals": '''Task:
                                            - - Step 1: Use ONLY the retrieved data below to construct a SINGLE counterfactual
                                            input vector for the query row.
                                            • For IMMUTABLE features
                                              {immutable_features}:
                                                - DO NOT change them.
                                                - Copy the original query row value exactly into "counterfactual_input".
                                            • For MUTABLE features
                                              {non_immutable_features}:
                                                - You may change these values to achieve the intended change
                                                  in LOS_CLASS (from Medium to Short).
                                            - Step 2: For EACH target in {target_list}, assign a separate set of feature importances
                                              across the input features {input_features}.
                                            - Step 3: Each target’s feature importances must be non-negative percentages that SUM TO 100 (per target).
                                            Step 4 (Local error, per target): For EACH target name T in {target_list},
                                                   find the columns "T_ground_truth" and "T_predicted" in the retrieved rows and compute
                                                   MAE[T] = mean( abs( row["T_ground_truth"] − row["T_predicted"] ) ) over all usable rows
                                                   (exclude missing/invalid; exclude the baseline row if instructed).
                                            - Return JSON ONLY matching the schema below. If sums differ due to rounding, renormalize to exactly 100.00.
                                        ''',
                    
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

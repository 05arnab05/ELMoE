# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:38:04 2025

@author: arnab
"""

Constraints_dict = {
                    "One_output":{
                                    "Accuracy":  '''CONSTRAINTS
                                                    - Output MUST be a single valid JSON object that EXACTLY matches the format shown (same keys, same casing).
                                                    - Use double quotes for all keys and string values.
                                                    - Do NOT add any extra keys, text, comments, explanations, code fences, or markdown.
                                                    - No trailing commas anywhere.
                                                    - Percentages MUST be numbers (not strings). If sums differ from 100 due to rounding, renormalize to sum exactly 100.00.
                                                    - "prediction" MUST contain exactly {Target_column_number} keys: {Target_column_names}.
                                                    - Start the response with "{{" and end with "}}".''',
                                    
                                    "Counterfactuals": '''CONSTRAINTS
                                                        - Output MUST be a single valid JSON object that EXACTLY matches the format shown (same keys, same casing).
                                                        - Keep Immutable feature values unchanged
                                                        - For each immutable feature
                                                          {immutable_features}:
                                                            • In "counterfactual_input" you MUST set the value to be EXACTLY the same
                                                              as in the original query row.
                                                            • You are NOT allowed to modify these values in any way (no small rounding changes).
                                                        - Use double quotes for all keys and string values.
                                                        - Do NOT add any extra keys, text, comments, explanations, code fences, or markdown.
                                                        - No trailing commas anywhere.
                                                        - "counterfactual_input" MUST contain exactly {Input_column_number} keys (the input features): {Input_column_names}.
                                                        - All immutable feature values MUST be unchanged and others must be numeric (int or float). Do NOT use strings, null, NaN, or scientific notation with quotes.
                                                        - Do NOT include any target/output columns; only input features are allowed.
                                                        - Start the response with "{{" and end with "}}".
                                                        - Try to bring the desired output change with minimum change in input fields
                                                        ''',
                                    
                                    "Feature_importance": ''''You will receive a dataframe of network features and predicted CQI labels. 
                                                        Your role is to quantify the importance of each feature in determining CQI using real numbers. 
                                                        Each feature should have a non-zero value. 
                                                        Use your own reasoning and do not include the target column CQI.''',
                                },
                    
                    "Two_output":{
                                    "Accuracy":  '''CONSTRAINTS
                                                    - Output MUST be a single valid JSON object that EXACTLY matches the format shown (same keys, same casing).
                                                    - Use double quotes for all keys and string values.
                                                    - Do NOT add any extra keys, text, comments, explanations, code fences, or markdown.
                                                    - No trailing commas anywhere.
                                                    - Percentages MUST be numbers (not strings). If sums differ from 100 due to rounding, renormalize to sum exactly 100.00.
                                                    - "prediction" MUST contain exactly {Target_column_number} keys: {Target_column_names}.
                                                    - "feature_importances" MUST contain exactly {Target_column_number} keys (one per target). Each target’s object MUST contain exactly {Input_column_number} keys: {Input_column_names}.
                                                    - Start the response with "{{" and end with "}}".''',
                                    
                                    "Counterfactuals": '''CONSTRAINTS
                                                        - Output MUST be a single valid JSON object that EXACTLY matches the format shown (same keys, same casing).
                                                        - Keep Immutable feature values unchanged
                                                        - For each immutable feature
                                                          {immutable_features}:
                                                            • In "counterfactual_input" you MUST set the value to be EXACTLY the same
                                                              as in the original query row.
                                                            • You are NOT allowed to modify these values in any way (no small rounding changes).
                                                            - For each mutable feature, you may change the value.
  
                                                        - Use double quotes for all keys and string values.
                                                        - Do NOT add any extra keys, text, comments, explanations, code fences, or markdown.
                                                        - No trailing commas anywhere.
                                                        - "counterfactual_input" MUST contain exactly {Input_column_number} keys (the input features): {Input_column_names}.
                                                        - All immutable feature values MUST be unchanged and others must be numeric (int or float). Do NOT use strings, null, NaN, or scientific notation with quotes.
                                                        - Do NOT include any target/output columns; only input features are allowed.
                                                        - If a feature should remain unchanged, copy the original query row value (do not omit).
                                                        - "feature_importances" MUST contain exactly {Target_column_number} keys (one per target). Each target’s object MUST contain exactly {Input_column_number} keys: {Input_column_names}.
                                                        - Start the response with "{{" and end with "}}".
                                                        - Try to bring the desired output change with minimum change in input fields
                                                        ''',
                                    
                                    "Feature_importance": ''''You will receive a dataframe of network features and predicted CQI labels. 
                                                        Your role is to quantify the importance of each feature in determining CQI using real numbers. 
                                                        Each feature should have a non-zero value. 
                                                        Use your own reasoning and do not include the target column CQI.''',
                                },
                    
                    "More_output":{
                                    "Accuracy":  '''CONSTRAINTS
                                                    - Output MUST be a single valid JSON object that EXACTLY matches the format shown (same keys, same casing).
                                                    - Use double quotes for all keys and string values.
                                                    - Do NOT add any extra keys, text, comments, explanations, code fences, or markdown.
                                                    - No trailing commas anywhere.
                                                    - Percentages MUST be numbers (not strings). If sums differ from 100 due to rounding, renormalize to sum exactly 100.00.
                                                    - "prediction" MUST contain exactly {Target_column_number} keys: {Target_column_names}.
                                                    - "feature_importances" MUST contain exactly {Target_column_number} keys (one per target). Each target’s object MUST contain exactly {Input_column_number} keys: {Input_column_names}.
                                                    - "local_error" MUST contain exactly one key "MAE" as an object with exactly {Target_column_number} keys (targets in {Target_column_names}), each mapped to a numeric value;
                                                    - Start the response with "{{" and end with "}}".''',
                                    
                                    "Counterfactuals": '''CONSTRAINTS
                                                        - Output MUST be a single valid JSON object that EXACTLY matches the format shown (same keys, same casing).
                                                        - Keep Immutable feature values unchanged
                                                        - For each immutable feature
                                                          {immutable_features}:
                                                            • In "counterfactual_input" you MUST set the value to be EXACTLY the same
                                                              as in the original query row.
                                                            • You are NOT allowed to modify these values in any way (no small rounding changes).
                                                        - Use double quotes for all keys and string values.
                                                        - Do NOT add any extra keys, text, comments, explanations, code fences, or markdown.
                                                        - No trailing commas anywhere.
                                                        - "counterfactual_input" MUST contain exactly {Input_column_number} keys (the input features): {Input_column_names}.
                                                        - All immutable feature values MUST be unchanged and others must be numeric (int or float). Do NOT use strings, null, NaN, or scientific notation with quotes.
                                                        - Do NOT include any target/output columns; only input features are allowed.
                                                        - If a feature should remain unchanged, copy the original query row value (do not omit).
                                                        - "feature_importances" MUST contain exactly {Target_column_number} keys (one per target). Each target’s object MUST contain exactly {Input_column_number} keys: {Input_column_names}.
                                                       - "local_error" MUST contain exactly one key "MAE" as an object with exactly {Target_column_number} keys (targets in {Target_column_names}), each mapped to a numeric value;
                                                        - Start the response with "{{" and end with "}}".
                                                        - Try to bring the desired output change with minimum change in input fields
                                                        ''',
                                    
                                    "Feature_importance": ''''You will receive a dataframe of network features and predicted CQI labels. 
                                                        Your role is to quantify the importance of each feature in determining CQI using real numbers. 
                                                        Each feature should have a non-zero value. 
                                                        Use your own reasoning and do not include the target column CQI.''',
                                }
                    
}
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 08:30:00 2025

@author: arnab
"""

import numpy as np
import pandas as pd


#for 1 decision tree
def extract_conditions(tree, data_point, feature_names):
    """
    Extract the conditions that lead to a specific output in a decision tree for a given data point.

    Parameters:
        tree: The decision tree model.
        data_point: A single data point to traverse the tree (1D array-like).
        feature_names: List of feature names corresponding to the features in the data.

    Returns:
        A string representation of the conditions and the output value.
    """
    # Access the tree structure
    tree_ = tree.tree_
    feature = tree_.feature
    threshold = tree_.threshold
    value = tree_.value

    # Traverse the path for this datapoint
    node = 0
    conditions = []
    while tree_.children_left[node] != -1:  # Not a leaf node
        feature_name = feature_names[feature[node]]
        if data_point[feature[node]] <= threshold[node]:
            conditions.append(f"{feature_name} <= {threshold[node]:.1f}")
            node = tree_.children_left[node]
        else:
            conditions.append(f"{feature_name} > {threshold[node]:.1f}")
            node = tree_.children_right[node]

    # Get the output value at the leaf node
    output_value = value[node]

    # Format the output as a string
    conditions_str = " AND ".join(conditions)
    output_str = f"{conditions_str}) => Output: {output_value}"

    return output_str


#Function to modify random forest
def aggregate_conditions(forest, data_point, feature_names):
    from collections import defaultdict

    def extract_conditions(tree, data_point, feature_names):
        # Access the tree structure
        tree_ = tree.tree_
        feature = tree_.feature
        threshold = tree_.threshold

        # Traverse the path for this datapoint
        node = 0
        conditions = []
        while tree_.children_left[node] != -1:  # Not a leaf node
            feature_name = feature_names[feature[node]]
            if data_point[feature[node]] <= threshold[node]:
                conditions.append((feature_name, "<=", threshold[node]))
                node = tree_.children_left[node]
            else:
                conditions.append((feature_name, ">", threshold[node]))
                node = tree_.children_right[node]
        return conditions

    # Aggregate conditions across all trees
    all_conditions = defaultdict(lambda: {"<=": [], ">": []})
    for tree in forest.estimators_:
        tree_conditions = extract_conditions(tree, data_point, feature_names)
        for feature_name, operator, threshold in tree_conditions:
            all_conditions[feature_name][operator].append(threshold)
    
    # Summarize conditions for each feature
    aggregated_conditions = []
    for feature_name, ops in all_conditions.items():
        avg_less_equal = np.mean(ops["<="]) if ops["<="] else None
        avg_greater = np.mean(ops[">"]) if ops[">"] else None

        if avg_less_equal is not None and (avg_greater is None or avg_less_equal <= avg_greater):
            aggregated_conditions.append(f"{feature_name} <= {avg_less_equal:.2f}")
        elif avg_greater is not None:
            aggregated_conditions.append(f"{feature_name} > {avg_greater:.2f}")
    
    # Combine into a single rule
    return " AND ".join(aggregated_conditions)

  
#for MLP and CNN

def TD_CSV_wide(X_train,Y_train,Y_predict,Batch_loss,epochs,batch_size):
    X_train=X_train.reset_index(drop=True)
    Y_train=Y_train.reset_index(drop=True)
    Y_predict=Y_predict.reset_index(drop=True)
    #Creating the loss df
    Batch_loss_df=Loss_df_wide(Batch_loss, batch_size, epochs)
    print(Batch_loss_df)
    #concat all df
    frames = [X_train, Y_train,Y_predict, Batch_loss_df]
    result = pd.concat(frames,axis=1)
    return result
        
        
        
def Loss_df_wide(Batch_loss,batch_size,epochs):
    
    #reshape in the form of training data
    Batch_loss_reshaped=np.array(Batch_loss).reshape(epochs, -1).T
    #expand because 10 in each batch
    Batch_loss_expanded=np.repeat(Batch_loss_reshaped, 10, axis=0)
    Batch_loss_df = pd.DataFrame(Batch_loss_expanded,columns=[f'epoch {i+1}' for i in range(epochs)])
    
    
    return Batch_loss_df
        
  

# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:38:04 2025

@author: arnab
"""
Model_description_dict = {
    "DT": ''' This is a Decision Tree regressor that uses the squared error criterion to measure the quality of splits. 
              It has no limit on tree depth or the number of leaf nodes, allowing it to grow fully until stopping conditions are met. 
              The tree uses the 'best' splitter strategy to choose the most optimal feature at each node. 
              It requires at least 2 samples to attempt a split and at least 1 sample per leaf. 
              No pruning is applied (ccp_alpha=0.0), and no feature or sample constraints are enforced during training. 
              The model is initialized with a fixed random state (random_state=42) for reproducibility and is well-suited for capturing non-linear relationships in tabular regression tasks."
                 ''',
    
    "RF": '''This is a Random Forest regressor consisting of 100 decision trees, using the squared error criterion to measure split quality. It employs bootstrap sampling (bootstrap=True) and uses all features when determining splits (max_features=1.0). 
            The model has no specified maximum tree depth, allowing trees to grow until all leaves are pure or contain fewer than the minimum required samples. 
            Each tree requires at least 2 samples to split and at least 1 sample per leaf. The model does not use out-of-bag evaluation (oob_score=False) and is initialized with a fixed random state (random_state=42) to ensure reproducibility. 
            It is well-suited for regression tasks on structured data with non-linear relationships.''',
    
    "MLP": '''The model named Multi-Layer Perceptron is a multi-layer perceptron designed for tabular data with 6 input features. 
               It consists of two hidden layers: the first with 64 neurons using ReLU activation, and the second with 32 neurons also using ReLU. 
               The output layer has 2 neurons with linear activation, suitable for regression tasks. All layers are trainable, and the model uses float32 precision with Glorot Uniform weight initialization. 
               This structure makes it well-suited for predicting two continuous values from structured input.''',
    
    "Transformer": '''The model named Feature Transformer is a Transformer encoder for tabular data with 8 input features. 
                  Each feature is projected to a 128-dimensional embedding with a learnable positional embedding to 
                  encode feature identity. The encoder stack has 2 layers, each with 4 self-attention heads and a 
                  256-unit feed-forward network. Encoded tokens are pooled, passed through a 64-unit ReLU layer, and 
                  mapped to 2 linear outputs for regression. All layers are trainable, using float32 precision, Glorot 
                  Uniform initialization, and the Adam optimizer. This design enables modeling of complex dependencies 
                  between input features.''',
}

Primary_features_dict = {
    "Radio and signal features": '''RSRQ (Reference Signal Received Quality): Indicates the quality of the received LTE signal by combining signal strength and interference; lower values suggest poor quality.
                                    RSSI (Received Signal Strength Indicator): Measures the total received power of the signal, including noise and interference; used to assess overall signal strength.
                 ''',
    
    "Location Features": '''Longitude: Geographic longitude (horizontal coordinate) of the user equipment or network event.
                            Latitude: Geographic latitude (vertical coordinate) of the user equipment or network event.''',
    
    " Channel Quality and Mobility": '''CQI (Channel Quality Indicator): Integer representing channel conditions as reported by the device; higher values imply better downlink channel quality.
              Speed: The velocity of the user equipment, typically in km/h or m/s; may affect signal stability and throughput.''',
    
    "Throughput Features": '''UL_bitrate (Uplink Bitrate): Actual uplink data transmission rate in bits per second.

                              DL_bitrate (Downlink Bitrate): Actual downlink data transmission rate in bits per second.

                              ('UL_bitrate_predicted',): Predicted uplink bitrate by the machine learning model, used to compare against actual performance.

                              ('DL_bitrate_predicted',): Predicted downlink bitrate by the machine learning model.''',
}


Secondary_features_dict = {
    "DT": '''DT_conditions: A string-encoded logical expression summarizing the combination of signal quality, geographic, and mobility conditions (e.g., RSRQ ≤ -0.62 AND Latitude > 0.37) used by the RF model to predict.
                 ''',
    
    "RF": '''RF_conditions: A string-encoded logical expression summarizing the combination of signal quality, geographic, and mobility conditions (e.g., RSRQ ≤ -0.62 AND Latitude > 0.37) used by the RF model to predict.''',
    
    "MLP": ''''Secondary features: epoch 1 to epoch 20: Model loss or error value after each training epoch (iteration over training data) for a row or batch of data; helps track learning progress.
               Cumulative_epoch_loss: The total or average loss accumulated across all epochs for a row or batch of data, used as a summary metric of model training quality.''',
               
    "Transformer": ''''Secondary features: epoch 1 to epoch 20: Model loss or error value after each training epoch (iteration over training data) for a row or batch of data; helps track learning progress.
               Cumulative_epoch_loss: The total or average loss accumulated across all epochs for a row or batch of data, used as a summary metric of model training quality.'''
}


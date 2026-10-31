#!/usr/bin/env python
# coding: utf-8

#%% In[49]:


import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#new
import sys
sys.path.append('C:/Users/arnab/Desktop/LLM black box model interface/python codes/Model metrics')


file_directory='C:/Users/arnab/Desktop/LLM black box model interface/files/'
folder_path =  "C:/Users/arnab/Desktop/LLM black box model interface/python codes/Yasith files/train 2"

csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
dataframes = [pd.read_csv(file) for file in csv_files]

combined_df = pd.concat(dataframes, ignore_index=True)

print("First few rows of the combined DataFrame:")
print(combined_df.head())

print("\nSummary Statistics (Min, Max, Average):")
summary_table = combined_df.describe().loc[['min', 'max', 'mean']].transpose()


summary_table.rename(columns={'mean': 'average'}, inplace=True)

print(summary_table)


#%% In[84]:


combined_df['RSRP'] = pd.to_numeric(combined_df['RSRP'], errors='coerce')
combined_df['RSRQ'] = pd.to_numeric(combined_df['RSRQ'], errors='coerce')
combined_df['SNR'] = pd.to_numeric(combined_df['SNR'], errors='coerce')
combined_df['RSSI'] = pd.to_numeric(combined_df['RSSI'], errors='coerce')
combined_df['CQI'] = pd.to_numeric(combined_df['CQI'], errors='coerce')
combined_df['DL_bitrate'] = pd.to_numeric(combined_df['DL_bitrate'], errors='coerce')
combined_df['UL_bitrate'] = pd.to_numeric(combined_df['UL_bitrate'], errors='coerce')
combined_df['Speed'] = pd.to_numeric(combined_df['Speed'], errors='coerce')
combined_df.dropna(subset=['RSRP', 'RSRQ', 'SNR', 'RSSI', 'CQI', 'DL_bitrate', 'UL_bitrate', 'Speed'], inplace=True)

X_labels=['RSRP', 'RSRQ', 'RSSI', 'Speed','DL_bitrate','UL_bitrate','SNR']
Y_labels=['CQI']

X = combined_df[X_labels].values
y = combined_df[Y_labels].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

poly = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)



combined_df.head()


#%% Decision Tree Regression Model




from sklearn.tree import DecisionTreeRegressor
from DT_model_files import DT_metrics

tree_model = DecisionTreeRegressor()
tree_model.fit(X_train, y_train) 

y_pred = tree_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Decision Tree Regression:")
print(f"  Mean Squared Error: {mse:.4f}")
print(f"  Mean Absolute Error: {mae:.4f}")
print(f"  R-squared: {r2:.4f}\n")

#Generating model files for DT
DT=DT_metrics(file_directory, tree_model)
DT.training_data(X_train, X_labels, y_train, Y_labels)
DT.test_data(X_test,X_labels, y_test,Y_labels)

# Random Forest Regressor

#%% Random forest regressor


from sklearn.ensemble import RandomForestRegressor
from RF_model_files import RF_metrics

tree_model = RandomForestRegressor()
tree_model.fit(X_train, y_train) 

y_pred = tree_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("RF Regression:")
print(f"  Mean Squared Error: {mse:.4f}")
print(f"  Mean Absolute Error: {mae:.4f}")
print(f"  R-squared: {r2:.4f}\n")

#Generating model files for RF
RF=RF_metrics(file_directory, tree_model)
RF.training_data(X_train, X_labels, y_train, Y_labels)
RF.test_data(X_test,X_labels, y_test,Y_labels)
# 

#%% Ridge Regression


from sklearn.linear_model import Ridge
from Ridge_model_files import Ridge_metrics

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

y_pred = ridge_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Ridge Regression:")
print(f"  Mean Squared Error: {mse:.4f}")
print(f"  Mean Absolute Error: {mae:.4f}")
print(f"  R-squared: {r2:.4f}\n")



#Generating model files for Ridge
Ridge=Ridge_metrics(file_directory,ridge_model)
Ridge.training_data(X_train, X_labels, y_train, Y_labels)
Ridge.test_data(X_test,X_labels, y_test,Y_labels)


#%% SVR Model


from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

svr_model = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)

X_train2= X_train_scaled[1:1000]
Y_train2= y_train[1:1000]
svr_model.fit(X_train_scaled, y_train)

y_pred_svr = svr_model.predict(X_test_scaled)

mse_svr = mean_squared_error(y_test, y_pred_svr)
mae = mean_absolute_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)

print('SVR Model:')
print(f'  Mean Squared Error: {mse_svr}')
print(f"  Mean Absolute Error: {mae:.4f}")
print(f'  R² Score: {r2_svr}')


#%% MLP Model

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


mlp_model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)  
])


mlp_model.compile(
    loss='mse',
    optimizer='adam',
    metrics=['mae']  
)


history_mlp = mlp_model.fit(
    X_train_scaled, y_train,
    validation_split=0.2, 
    epochs=10,
    batch_size=16,
    verbose=1
)


y_pred_mlp = mlp_model.predict(X_test_scaled).ravel()  

mse_mlp = mean_squared_error(y_test, y_pred_mlp)
mae_mlp = mean_absolute_error(y_test, y_pred_mlp)
r2_mlp = r2_score(y_test, y_pred_mlp)

print("MLP Regression:")
print(f"  Mean Squared Error: {mse_mlp:.4f}")
print(f"  Mean Absolute Error: {mae_mlp:.4f}")
print(f"  R-squared: {r2_mlp:.4f}\n")


#%% CNN Model

import numpy as np
from tensorflow.keras.layers import Conv1D, Flatten, Dense


X_train_cnn = np.reshape(X_train_scaled, (X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_cnn = np.reshape(X_test_scaled, (X_test_scaled.shape[0], X_test_scaled.shape[1], 1))


cnn_model = keras.Sequential([
    Conv1D(filters=16, kernel_size=2, activation='relu', input_shape=(X_train_cnn.shape[1], X_train_cnn.shape[2])),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)  
])

cnn_model.compile(
    loss='mse',
    optimizer='adam',
    metrics=['mae']
)

history_cnn = cnn_model.fit(
    X_train_cnn, y_train,
    validation_split=0.2,  
    epochs=100,
    batch_size=16,
    verbose=1
)

y_pred_cnn = cnn_model.predict(X_test_cnn).ravel()

mse_cnn = mean_squared_error(y_test, y_pred_cnn)
mae_cnn = mean_absolute_error(y_test, y_pred_cnn)
r2_cnn = r2_score(y_test, y_pred_cnn)

print("CNN Regression (Scaled Inputs):")
print(f"  Mean Squared Error: {mse_cnn:.4f}")
print(f"  Mean Absolute Error: {mae_cnn:.4f}")
print(f"  R-squared: {r2_cnn:.4f}\n")


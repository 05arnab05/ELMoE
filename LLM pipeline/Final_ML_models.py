
# Init everything


import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from Custom_transformer_layers_and_helper_functions import FeaturePositionalEmbedding

variables='Three variables' #'Two variables' 'Three variables'
model_file_generation= False

import sys
sys.path.append('C:/Users/arnab/Desktop/LLM black box model interface/python codes/Model metrics_file_generator')

file_directory=f'C:/Users/arnab/Desktop/LLM black box model interface/files_trial_new/{variables}/'
folder_path =  "C:/Users/arnab/Desktop/LLM black box model interface/python codes/Yasith files/datasets/static"


csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
dataframes = [pd.read_csv(file) for file in csv_files]

combined_df = pd.concat(dataframes, ignore_index=True)

#1 variable: DL_bitrate
#2 variable: DL_bitrate and RSRP
#3 variables: DL_bitrate, RSRP and SNR


cols = ['RSRP', 'RSRQ', 'SNR', 'RSSI', 
        'DL_bitrate', 'UL_bitrate', 'Longitude', 'Latitude', 'Speed']
for col in cols:
    combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
combined_df.dropna(subset=cols, inplace=True)


#1 variables
if(variables=='One variable'):
    features = ['RSRQ', 'RSSI', 'Longitude', 'Latitude', 'CQI','Speed','UL_bitrate', 'SNR','RSRP' ]
    target_cols = ['DL_bitrate']
    


#2 variables
if(variables=='Two variables'):
    features = ['RSRQ', 'RSSI', 'Longitude', 'Latitude', 'CQI','Speed','UL_bitrate', 'SNR']
    target_cols = ['DL_bitrate', 'RSRP']



#3 variables
if(variables=='Three variables'):
    features = ['RSRQ', 'RSSI', 'Longitude', 'Latitude', 'CQI','Speed','UL_bitrate']
    target_cols = ['DL_bitrate','RSRP', 'SNR']




X = combined_df[features].values
y = combined_df[target_cols].values  

X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
Y_train_scaled = scaler_y.fit_transform(Y_train)
Y_test_scaled = scaler_y.transform(Y_test)

def evaluate_performance(y_true, y_pred, target_names):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape, f"Shape mismatch: {y_true.shape} vs {y_pred.shape}"

    # Per-target RMSE and MAE
    mse_vec = mean_squared_error(y_true, y_pred, multioutput='raw_values')
    rmse_vec = np.sqrt(mse_vec)
    mae_vec  = mean_absolute_error(y_true, y_pred, multioutput='raw_values')

    for i, target in enumerate(target_names):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        # Baseline: mean of y_true
        baseline_pred = np.full_like(y_t, np.mean(y_t))
        baseline_rmse = np.sqrt(np.mean((y_t - baseline_pred) ** 2))

        # Relative improvement over baseline RMSE (1 = perfect, 0 = baseline, negative = worse)
        rmse_improvement = 1 - (rmse_vec[i] / baseline_rmse) if baseline_rmse != 0 else np.nan

        # Symmetric MAPE
        denom = (np.abs(y_t) + np.abs(y_p)) / 2
        smape = np.mean(
            np.where(denom == 0, 0.0, np.abs(y_p - y_t) / denom)
        ) * 100

        print(f"Target: {target}")
        print(f"  RMSE           : {rmse_vec[i]:.4f}")
        print(f"  MAE            : {mae_vec[i]:.4f}")
        print(f"  RMSE Improvement over baseline: {rmse_improvement:.4f}")
        print(f"  sMAPE (%)      : {smape:.2f}")
    print()

def Inverse_Scaling(Scaling_Y,y_prediction):
    Unscaled_predictions=Scaling_Y.inverse_transform(y_prediction.reshape(-1,len(target_cols)))
    return Unscaled_predictions
    


#%% Decision Tree


from sklearn.tree import DecisionTreeRegressor
from DT_model_files import DT_metrics

dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, Y_train)

y_pred_dt = dt_model.predict(X_test)
y_pred_dt_reshaped=y_pred_dt.reshape(-1,len(target_cols))
print("Decision Tree Regression Performance:")
evaluate_performance(Y_test, y_pred_dt_reshaped, target_cols)

if model_file_generation:
    DT=DT_metrics(file_directory, dt_model)
    td_DT=DT.training_data(X_train, features, Y_train, target_cols)
    DT.test_data(X_test,features, Y_test,target_cols)



# %%Ridge regression


from sklearn.linear_model import Ridge
from Ridge_model_files import Ridge_metrics

ridge_model = Ridge()
ridge_model.fit(X_train_scaled, Y_train_scaled)

y_pred_ridge = ridge_model.predict(X_test_scaled)
Y_pred_ridge_unscaled=Inverse_Scaling(scaler_y, y_pred_ridge)
print("Ridge Regression Performance:")
evaluate_performance(Y_test, Y_pred_ridge_unscaled, target_cols)

if model_file_generation:
    Ridge_obj=Ridge_metrics(file_directory, ridge_model)
    td_Ridge=Ridge_obj.training_data(X_train, features, Y_train, target_cols)
    Ridge_obj.test_data(X_test,features, Y_test,target_cols)

    


#%% RF
from sklearn.ensemble import RandomForestRegressor
from RF_model_files import RF_metrics

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)

y_pred_rf = rf_model.predict(X_test)
y_pred_rf_reshaped=y_pred_rf.reshape(-1,len(target_cols))
print("Random Forest Regression Performance:")
evaluate_performance(Y_test, y_pred_rf_reshaped, target_cols)

#Generating model files for RF
if model_file_generation:
    RF=RF_metrics(file_directory, rf_model)
    td_RF=RF.training_data(X_train, features, Y_train, target_cols)
    RF.test_data(X_test,features, Y_test,target_cols)
     


#%% MLP


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from MLP_model_files import MLP_metrics,LossHistory


mlp_model = keras.Sequential([  
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(len(target_cols), activation='linear')  
])

mlp_model.compile(
    loss='mean_squared_error',
    optimizer='adam'
)
history = LossHistory()

mlp_model.fit(X_train_scaled, Y_train_scaled, epochs=20, batch_size=32, verbose=0,callbacks=[history])
loss=history.losses

y_pred_mlp = mlp_model.predict(X_test_scaled)
Y_pred_mlp_unscaled=Inverse_Scaling(scaler_y, y_pred_mlp)
print("Keras MLP Performance:")
evaluate_performance(Y_test, Y_pred_mlp_unscaled, target_cols)

batch_size=32
epochs=20
batch_loss_list=[round(num, 2) for num in loss]

if model_file_generation:
    MLP_obj=MLP_metrics(file_directory, mlp_model)
    td_mlp=MLP_obj.training_data(X_train, features, Y_train, target_cols, batch_loss_list,epochs,batch_size)
    MLP_obj.test_data(X_test,features, Y_test,target_cols)







#%% CNN 

#MAYBE

# from CNN_model_files import CNN_metrics, LossHistory
# from tensorflow import keras
# from tensorflow.keras import layers

# # --- reshape: (n_samples, n_features) -> (n_samples, n_features, 1) ---
# X_train_cnn = X_train_scaled.astype("float32").reshape(-1, X_train_scaled.shape[1], 1)
# X_test_cnn  = X_test_scaled.astype("float32").reshape(-1,  X_test_scaled.shape[1],  1)

# n_features = X_train_cnn.shape[1]
# n_targets  = len(target_cols)

# # --- Conv1D model with kernel_size=1 (treat each sample independently) ---
# cnn_model = keras.Sequential([
#     layers.Input(shape=(n_features, 1)),
#     layers.Conv1D(64, kernel_size=1, activation='relu'),
#     layers.Conv1D(32, kernel_size=1, activation='relu'),
#     layers.GlobalAveragePooling1D(),                 # aggregates over feature positions
#     layers.Dense(n_targets, activation='linear'),
# ], name="CNN1D_featurewise")

# cnn_model.compile(loss='mean_squared_error', optimizer='adam')
# history = LossHistory()

# cnn_model.fit(X_train_cnn, Y_train_scaled, epochs=20, batch_size=32, verbose=0, callbacks=[history])
# loss = history.losses

# # --- Predict & inverse-scale ---
# y_pred_cnn_scaled = cnn_model.predict(X_test_cnn, verbose=0)
# Y_pred_cnn = Inverse_Scaling(scaler_y, y_pred_cnn_scaled)

# print("Keras CNN(1D, k=1) Performance:")
# evaluate_performance(Y_test, Y_pred_cnn, target_cols)

# # --- bookkeeping / saving, mirroring your MLP flow ---
# batch_size = 32
# epochs = 20
# batch_loss_list = [round(num, 2) for num in loss]

# # CNN_obj = CNN_metrics(file_directory, cnn_model)  # reuse your wrapper if it just needs (dir, model)
# # td = CNN_obj.training_data(X_train, features, Y_train, target_cols,  batch_loss_list, epochs, batch_size)
# # CNN_obj.test_data(X_test, features, Y_test, target_cols)


#%% transformer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from Transformer_model_files import Transformer_metrics,LossHistory

def transformer_encoder(d_model, num_heads, d_ff, dropout, count):
    
    inputs = layers.Input(shape=(None, d_model))      # (batch, F, d_model)
    x = inputs

    attn = layers.MultiHeadAttention(num_heads=num_heads,
                                     key_dim=d_model // num_heads,
                                     name=f"mha_{count}")(x, x)
    x = layers.Add(name=f"res_attn_{count}")([x, attn])
    x = layers.LayerNormalization(epsilon=1e-6, name=f"ln1_{count}")(x)
    x = layers.Dropout(dropout, name=f"drop_attn_{count}")(x)

    ff = layers.Dense(d_ff, activation="relu", name=f"ff1_{count}")(x)
    ff = layers.Dense(d_model, name=f"ff2_{count}")(ff)
    x = layers.Add(name=f"res_ff_{count}")([x, ff])
    x = layers.LayerNormalization(epsilon=1e-6, name=f"ln2_{count}")(x)
    x = layers.Dropout(dropout, name=f"drop_ff_{count}")(x)

    return keras.Model(inputs, x, name=f"TransformerEncoderBlock{count}")

# --- Factory function ---

#%% transformer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from Transformer_model_files import Transformer_metrics,LossHistory

def transformer_encoder(d_model, num_heads, d_ff, dropout, count):
    
    inputs = layers.Input(shape=(None, d_model))      # (batch, F, d_model)
    x = inputs

    attn = layers.MultiHeadAttention(num_heads=num_heads,
                                     key_dim=d_model // num_heads,
                                     name=f"mha_{count}")(x, x)
    x = layers.Add(name=f"res_attn_{count}")([x, attn])
    x = layers.LayerNormalization(epsilon=1e-6, name=f"ln1_{count}")(x)
    x = layers.Dropout(dropout, name=f"drop_attn_{count}")(x)

    ff = layers.Dense(d_ff, activation="relu", name=f"ff1_{count}")(x)
    ff = layers.Dense(d_model, name=f"ff2_{count}")(ff)
    x = layers.Add(name=f"res_ff_{count}")([x, ff])
    x = layers.LayerNormalization(epsilon=1e-6, name=f"ln2_{count}")(x)
    x = layers.Dropout(dropout, name=f"drop_ff_{count}")(x)

    return keras.Model(inputs, x, name=f"TransformerEncoderBlock{count}")

# --- Factory function ---

def build_feature_transformer(
    num_features: int,
    num_outputs: int,
    d_model: int = 128,
    num_heads: int = 4,
    d_ff: int = 256,
    depth: int = 2,
    dropout: float = 0.1,
    lr: float = 1e-3,
    name: str = "FeatureTransformerRegressor",
):
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    
    #input layer all input features of our predictor
    inputs = keras.Input(shape=(num_features,), name="features_scalar")   # (batch, F)

    # tokeniser, converts each input to a token
    x = layers.Reshape((num_features, 1), name="to_tokens")(inputs)       # (batch, F, 1)
    
    #simple dense layer which projects each of our n features to a d_model space (simple feed forward decoder)
    x = layers.Dense(d_model, name="feature_projection")(x)               # (batch, F, d_model)

    # Add learnable FEATURE positional embedding (0..F-1) basically an F by F 2D matrix
    x = FeaturePositionalEmbedding(num_features=num_features, d_model=d_model, name="feat_pos_emb")(x)

    # Encoder stack
    for i in range(depth):
        x = transformer_encoder(d_model, num_heads, d_ff, dropout, i)(x)

    # Pool tokens and predict
    x = layers.GlobalAveragePooling1D(name="pool_features")(x)
    x = layers.Dense(64, activation="relu", name="head_hidden")(x)
    outputs = layers.Dense(num_outputs, activation="linear", name="pred")(x)

    model = keras.Model(inputs, outputs, name=name)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mean_squared_error")
    return model


F = len(features)
O = len(target_cols)

transformer_model = build_feature_transformer(num_features=F, num_outputs=O,
                                  d_model=128, num_heads=4, d_ff=256,
                                  depth=2, dropout=0.1, lr=1e-3)


history = LossHistory()
epochs=20
batch_size=32
transformer_model.fit(X_train_scaled, Y_train_scaled, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[history])
loss = history.losses

y_pred_scaled = transformer_model.predict(X_test_scaled, verbose=0)

Y_pred_unscaled = scaler_y.inverse_transform(y_pred_scaled)
print("Feature-Transformer Performance:")
evaluate_performance(Y_test, Y_pred_unscaled, target_cols)

batch_loss_list = [round(num, 2) for num in loss]


if model_file_generation:
    transformer_obj = Transformer_metrics(file_directory, transformer_model)
    td_tfm = transformer_obj.training_data(X_train, features, Y_train, target_cols, batch_loss_list, epochs, batch_size)
    transformer_obj.test_data(X_test, features, Y_test, target_cols)
    



import numpy as np
import pandas as pd
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import joblib

import sys
path_to_add = r"C:/Users/arnab/Desktop/LLM black box model interface/python codes/LLM pipeline"
sys.path.append(path_to_add)

from Cat_variable_functions import fit_cat_encoder, transform_cat_encoder

# -----------------------
# CONFIG
# -----------------------

variables = 'One variable'
model_file_generation = True

save_file_directory = f'C:/Users/arnab/Desktop/LLM black box model interface/files_trial_new/Healthcare/{variables}/'
dataset_folder_path = Path("C:/Users/arnab/Desktop/LLM black box model interface/Datasets/Healthcare/MIMIC 3 full/")

# -----------------------
# LOAD DATA
# -----------------------
df = pd.read_csv(dataset_folder_path /"mimic3_classification_ready_vitals_labs.csv")

print("✅ Loaded dataset:", df.shape)
print("Columns:", list(df.columns)[:15])

# -----------------------
# DEFINE TARGET: LOS_CLASS (categorical length-of-stay)
# -----------------------
los_clean = df["LOS_DAYS"].clip(lower=0, upper=60)
q = los_clean.quantile([0.33, 0.66])
q1, q2 = q[0.33], q[0.66]

bins = [-np.inf, q1, q2, np.inf]
labels = ["Short", "Medium", "Long"]
mapping = {0: 'short', 1: 'medium', 2: 'long'}
df["LOS_CLASS"] = pd.cut(df["LOS_DAYS"], bins=bins, labels=labels)
df = df.dropna(subset=["LOS_CLASS"])

# -----------------------
# DEFINE FEATURES AND TARGET
# -----------------------
drop_cols = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "LOS_DAYS", "LOS_CLASS"]
target_cols = ["LOS_CLASS"]

# Remove columns starting with LOG
cols_starting_with_LOG = [c for c in df.columns if c.startswith("LOG")]

# Combine all columns to drop
all_drop_cols = drop_cols + cols_starting_with_LOG

# All other columns except IDs, LOS, and LOG* are features
features = [c for c in df.columns if c not in all_drop_cols]

X = df[features]
y = df[target_cols]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cat_cols =  X.select_dtypes(include=["object", "category"]).columns
num_cols = [c for c in X.columns if c not in cat_cols]

X_train_num = X_train[num_cols].astype(float)
X_test_num  = X_test[num_cols].astype(float)
X_train_cat = X_train[cat_cols]
X_test_cat  = X_test[cat_cols]

# -----------------------
# Scale X numerical variables
# -----------------------
X_num_scaler = StandardScaler()
X_train_num_scaled = X_num_scaler.fit_transform(X_train_num)
X_test_num_scaled  = X_num_scaler.transform(X_test_num)

# -----------------------
# ENCODE X categorical variables
# -----------------------

# categorical integer encoding (no scaling)
X_train_cat_enc, X_encoder = fit_cat_encoder(X_train_cat)
X_test_cat_enc = transform_cat_encoder(X_test_cat, X_encoder)

# Save X encoders + scaler for inference later
joblib.dump(X_num_scaler, os.path.join(save_file_directory, "X_num_scaler.pkl"))
joblib.dump(X_encoder, os.path.join(save_file_directory, "X_cat_encoders.pkl"))


#Ridge special input
X_train_combined = pd.concat([X_train_num, X_train_cat_enc], axis=1)
X_test_combined=pd.concat([X_test_num, X_test_cat_enc], axis=1)

# -----------------------
# ENCODE TARGET
# -----------------------
Y_train_enc, Y_encoder = fit_cat_encoder(Y_train)
Y_test_enc = transform_cat_encoder(Y_test, Y_encoder)

#Saving encoders Y
Y_encoder_path = os.path.join(save_file_directory, "Ylabel_encoders.pkl")
joblib.dump(Y_encoder, Y_encoder_path)

# Scaled compild X train and test
X_train_enc_unscaled = np.hstack([X_train_num, X_train_cat_enc.values])
X_test_enc_unscaled  = np.hstack([X_test_num,  X_test_cat_enc.values])

# Scaled compild X train and test
X_train_enc_scaled = np.hstack([X_train_num_scaled, X_train_cat_enc.values])
X_test_enc_scaled  = np.hstack([X_test_num_scaled,  X_test_cat_enc.values])

# Combined feature names (numeric first, then categorical)
feature_names_num = list(num_cols)
feature_names_cat = list(X_train_cat_enc.columns)
all_feature_names = feature_names_num + feature_names_cat

for col in target_cols:
    target_le = Y_encoder[col]
    class_names = list(target_le.classes_)
    print(col, class_names)



# -----------------------
# EVALUATION FUNCTION
# -----------------------
def evaluate_classification(y_true, y_pred, class_names):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    print("Accuracy:", round(acc, 4))
    print("Macro-F1:", round(f1, 4))
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))



print("Class distribution (raw):")
print(df["LOS_CLASS"].value_counts())
print("\nClass distribution (%):")
print(df["LOS_CLASS"].value_counts(normalize=True).round(3))

#%% 🌳 Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
from DT_model_files import DT_metrics

dt_model = DecisionTreeClassifier(max_depth=6, random_state=42)
dt_model.fit(X_train_enc_unscaled, Y_train_enc)
y_pred_dt = dt_model.predict(X_test_enc_unscaled)

print("\n🌳 Decision Tree Classification Performance:")
evaluate_classification(Y_test_enc, y_pred_dt, class_names)


if model_file_generation:
    DT=DT_metrics(save_file_directory, dt_model)
    td_DT=DT.training_data(X_train_enc_unscaled, features, Y_train, target_cols, skip_secondary_features=True, classification=True, classification_encoder=Y_encoder, X_OG=X_train)
    DT.test_data(X_test,features, Y_test, target_cols)
    
#%% ⚙️ Ridge Classifier
from sklearn.linear_model import RidgeClassifier
path_to_add = r"C:/Users/arnab/Desktop/LLM black box model interface/python codes/LLM pipeline/Model metrics_file_generator"
sys.path.append(path_to_add)
from Ridge_model_files import Ridge_metrics

ridge_model = RidgeClassifier(alpha=1.0)
ridge_model.fit(X_train_enc_scaled, Y_train_enc)
y_pred_ridge = ridge_model.predict(X_test_enc_scaled)

print("\n⚙️ Ridge Classifier Performance:")
evaluate_classification(Y_test_enc, y_pred_ridge, class_names)

if model_file_generation:
    Ridge_obj=Ridge_metrics(save_file_directory, ridge_model)
    td_Ridge=Ridge_obj.training_data(X_train, features, Y_train, target_cols, classification=True, X_encoder=X_encoder,Y_encoder=Y_encoder,scaler_X=X_num_scaler, X_OG=X_train)
    Ridge_obj.test_data(X_test,features, Y_test,target_cols)

#%% 🌲 Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from RF_model_files import RF_metrics

rf_model = RandomForestClassifier(n_estimators=300, random_state=42)
rf_model.fit(X_train_enc_unscaled, Y_train_enc)
y_pred_rf = rf_model.predict(X_test_enc_unscaled)

print("\n🌲 Random Forest Classification Performance:")
evaluate_classification(Y_test_enc, y_pred_rf, class_names)

#Generating model files for RF
if model_file_generation:
    RF=RF_metrics(save_file_directory, rf_model)
    td_RF=RF.training_data(X_train_enc_unscaled, features, Y_train, target_cols, skip_secondary_features=True, classification=True, classification_encoder=Y_encoder, X_OG=X_train)
    RF.test_data(X_test,features, Y_test,target_cols)

#%%
from xgboost import XGBClassifier

# XGBoost generally prefers unscaled features (like RF), so use X_train / X_test
xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",   # multi-class, probability output
    eval_metric="mlogloss",
    num_class=len(class_names),
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, Y_train_enc)

y_pred_xgb = xgb_model.predict(X_test)

print("\n🚀 XGBoost Classification Performance:")
evaluate_classification(Y_test_enc, y_pred_xgb, class_names)

#%% Keras neural net data prep
# One target column name
target_col = target_cols[0]  # "LOS_CLASS"

# 1D arrays for Keras & class_weight
y_train_enc_1d = Y_train_enc[target_col].astype(int).to_numpy()  # shape (N,)
y_test_enc_1d  = Y_test_enc[target_col].astype(int).to_numpy()   # shape (N,)


# For Keras: numeric input is (N, num_numeric_features)
X_train_num_input = X_train_num_scaled
X_test_num_input  = X_test_num_scaled

# For Keras: each categorical column is (N, 1) int32
cat_cols = list(X_train_cat_enc.columns)  # keep order explicit
X_train_cat_inputs = [
    X_train_cat_enc[col].astype("int32").values.reshape(-1, 1)
    for col in cat_cols
]
X_test_cat_inputs = [
    X_test_cat_enc[col].astype("int32").values.reshape(-1, 1)
    for col in cat_cols
]

# Full input lists for .fit() / .predict()
X_train_inputs = [X_train_num_input] + X_train_cat_inputs
X_test_inputs  = [X_test_num_input]  + X_test_cat_inputs

target_col = target_cols[0]
target_le = Y_encoder[target_col]
class_names = list(target_le.classes_)
n_classes = len(class_names)

#%% MLP model preparation

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
path_to_add = r"C:/Users/arnab/Desktop/LLM black box model interface/python codes/LLM pipeline/Model metrics_file_generator"
sys.path.append(path_to_add)
from MLP_model_files import MLP_metrics, LossHistory

# Hyperparameters
batch_size = 32
epochs = 5
history = LossHistory()

# ---------- Inputs ----------
num_input = Input(shape=(X_train_num_input.shape[1],), name="numeric_input")

cat_inputs = []
cat_embeds = []

for col in cat_cols:
    #input to the embedding layer
    n_cat = X_encoder[col].classes_.shape[0]      # number of unique categories
    emb_dim = min(50, (n_cat // 2) + 1)          # simple rule-of-thumb

    inp = Input(shape=(1,), name=f"cat_{col}")
    #Initialize embedding layer
    emb = Embedding(input_dim=n_cat,
                    output_dim=emb_dim,
                    name=f"emb_{col}")(inp)
    emb = Flatten()(emb)

    cat_inputs.append(inp)
    cat_embeds.append(emb)

# Concatenate numeric + all embeddings
x = Concatenate(name="concat_features")([num_input] + cat_embeds)

# ---------- MLP body ----------
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
# you can add Dropout(...) if you like

output = Dense(n_classes, activation='softmax', name="output")(x)

# Complete model
mlp_model = Model(
    inputs=[num_input] + cat_inputs,
    outputs=output
)

mlp_model.compile(
    optimizer=Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

mlp_model.summary()
#%%MLP model fit and save

history = LossHistory()

mlp_obj = mlp_model.fit(
    X_train_inputs,
    y_train_enc_1d,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2,
    callbacks=[history],
    verbose=1
)

y_pred_probs = mlp_model.predict(X_test_inputs)
y_pred_mlp = np.argmax(y_pred_probs, axis=1)

print("\n🧠 MLP Classifier Performance:")
evaluate_classification(y_test_enc_1d, y_pred_mlp, class_names)


loss=history.losses
batch_loss_list=[round(num, 2) for num in loss]

if model_file_generation:
    MLP_obj=MLP_metrics(save_file_directory, mlp_model)
    td_mlp=MLP_obj.training_data(X_train, features, Y_train, target_cols, batch_loss_list,  epochs, batch_size, 
                                 scaler_X=X_num_scaler, scaler_Y=None,
                                 skip_secondary_features=True,
                                 classification=True, X_CF_encoder=X_encoder,Y_CF_encoder=Y_encoder, X_OG=X_train)
    MLP_obj.test_data(X_test,features, Y_test,target_cols)


#%%FTTransformer

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys
sys.path.append("C:/Users/arnab/Desktop/LLM black box model interface/python codes/LLM pipeline/")

from FTransformer_model_classifier import build_feature_transformer_with_cats
from Transformer_model_files import Transformer_metrics, LossHistory

# -----------------------
# CONFIGURATION
# -----------------------
num_numeric = X_train_num_input.shape[1]
cat_cardinalities = [X_encoder[col].classes_.shape[0] for col in cat_cols]
O = n_classes  # len(class_names)

transformer_model = build_feature_transformer_with_cats(
    num_numeric_features=num_numeric,
    cat_cardinalities=cat_cardinalities,
    num_outputs=O,
    d_model=128,
    num_heads=4,
    d_ff=256,
    depth=2,
    dropout=0.1,
    lr=1e-3,
    task="classification",
    name="FTTransformer_LOS_with_cats"
)

# -----------------------
# CLASS WEIGHTS  ✅ use 1D labels
# -----------------------
from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(y_train_enc_1d)
cw = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train_enc_1d          # <-- FIX HERE
)
class_weights = {int(c): float(w) for c, w in zip(classes, cw)}
print("Class weights:", class_weights)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=6,
    restore_best_weights=True
)
reduce_lr  = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-5
)

# -----------------------
# TRAIN  ✅ use embedded inputs + 1D y
# -----------------------
history = transformer_model.fit(
    X_train_inputs,        # [X_train_num_input] + X_train_cat_inputs
    y_train_enc_1d,        # 1D labels
    validation_split=0.1,
    epochs=10,
    batch_size=64,
    verbose=1,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr]
)

# -----------------------
# EVAL  ✅ use embedded inputs + 1D y_test
# -----------------------
y_pred_probs = transformer_model.predict(X_test_inputs, verbose=0)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

print("\n🧩 FTTransformer Classification Performance:")
evaluate_classification(y_test_enc_1d, y_pred_classes, class_names)




# -----------------------
# OPTIONAL: SAVE MODEL + TRAINING LOG
# -----------------------
if model_file_generation:
    transformer_obj = Transformer_metrics(save_file_directory, transformer_model)
    td_tfm = transformer_obj.training_data(X_train, features, Y_train, target_cols, batch_loss_list,  epochs, batch_size, 
                                 scaler_X=X_num_scaler, scaler_Y=None,
                                 skip_secondary_features=True,
                                 classification=True, X_CF_encoder=X_encoder,Y_CF_encoder=Y_encoder, X_OG=X_train)
    transformer_obj.test_data(X_test, features, Y_test, target_cols)
    

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





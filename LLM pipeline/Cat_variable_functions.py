# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 12:13:10 2025

@author: arnab


"""
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

def encode_query_classification(query):
    #Convert string->df
    
    #Convert string → dict
    parts = query.split(',')
    data = {}
    for part in parts:
        k, v = part.split(':', 1)
        data[k.strip()] = v.strip()
    #Convert dict → DataFrame (1 row)
    q_df = pd.DataFrame([data])
    
    return q_df
def fit_cat_encoder(df):
    encoders = {}
    df_encoded = df.copy()

    for col in df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df_encoded, encoders


def transform_cat_encoder(df, encoders):
    df_encoded = df.copy()

    for enc_key, le in encoders.items():
        # apply encoder to any column that starts with this key
        for col in df.columns:
            if col.startswith(enc_key):
                df_encoded[col] = le.transform(df[col].values)

    return df_encoded

def Encode_all(df, X_encoders=None, Y_encoders=None):
    """
    Apply categorical encoders to columns in df.

    - X_encoders: dict mapping prefix -> encoder
      Encoder applied to any column whose name starts with that prefix.

    - Y_encoders: dict mapping prefix -> encoder
      Encoder applied to any column whose name starts with that prefix.
    """
    df_encoded = df.copy()

    # Apply X encoders (prefix-based)
    if X_encoders is not None:
        for enc_key, le in X_encoders.items():
            for col in df.columns:
                if col.startswith(enc_key):
                    df_encoded[col] = le.transform(df[col].astype(str).values)

    # Apply Y encoders (prefix-based)
    if Y_encoders is not None:
        for enc_key, le in Y_encoders.items():
            for col in df.columns:
                if col.startswith(enc_key):
                    df_encoded[col] = le.transform(df[col].astype(str).values)

    return df_encoded

def Decode_all(df, X_encoders=None, Y_encoders=None):
    """
    Apply inverse categorical encoders to columns in df.

    - X_encoders: dict mapping prefix -> encoder
      Decoder applied to any column whose name starts with that prefix.

    - Y_encoders: dict mapping prefix -> encoder
      Decoder applied to any column whose name starts with that prefix.
    """
    df_decoded = df.copy()

    # Apply X decoders (prefix-based)
    if X_encoders is not None:
        for enc_key, le in X_encoders.items():
            for col in df.columns:
                if col.startswith(enc_key):
                    df_decoded[col] = le.inverse_transform(df[col].astype(int).values)

    # Apply Y decoders (prefix-based)
    if Y_encoders is not None:
        for enc_key, le in Y_encoders.items():
            for col in df.columns:
                if col.startswith(enc_key):
                    df_decoded[col] = le.inverse_transform(df[col].astype(int).values)

    return df_decoded


def decode_target_column(df, predicted_col, encoder_dict):
    """
    Decode a predicted column whose name may contain prefixes/suffixes 
    (e.g., 'LOS_CLASS_predicted') by matching it to the encoder dict keys 
    using startswith logic.

    df : DataFrame containing the column to decode
    predicted_col : the column name in df (e.g., "LOS_CLASS_predicted")
    encoder_dict : dict mapping original column names -> LabelEncoder
    """
    
    # Find key in encoder_dict that matches the start of predicted_col
    match = None
    for key in encoder_dict.keys():
        if predicted_col.startswith(key):
            match = key
            break

    if match is None:
        raise ValueError(f"No encoder found for column '{predicted_col}'. "
                         f"Available: {list(encoder_dict.keys())}")

    le = encoder_dict[match]

    # Decode the values
    df[predicted_col] = le.inverse_transform(df[predicted_col].astype(int))

    return df

import numpy as np
import pandas as pd

def Ridge_prepare_inputs_from_df(
    X_new,
    num_scaler,
    cat_encoders,
    feature_names=None,
    Only_scaling=False,
    unknown_value=-1
):
    """
    Convert X_new into the Ridge input matrix:
      X_enc_scaled = hstack([scaled_numeric, encoded_categoricals])

    - cat_encoders: dict {col_name: fitted LabelEncoder-like object}
    - num_scaler: fitted StandardScaler for numeric columns
    - Only_scaling:
        False -> assume categoricals are raw strings and must be encoded
        True  -> assume categoricals are already encoded ints (no encoding)
    - unknown_value: value to use for unseen categories
    """

    # ---- Convert ndarray -> DataFrame if needed
    if isinstance(X_new, np.ndarray):
        if feature_names is None:
            raise ValueError("feature_names must be provided when X_new is a NumPy array.")
        X_new = np.atleast_2d(X_new)
        if len(feature_names) != X_new.shape[1]:
            raise ValueError(
                f"Number of feature names ({len(feature_names)}) "
                f"does not match number of columns ({X_new.shape[1]})."
            )
        X_new = pd.DataFrame(X_new, columns=feature_names)

    # ---- Determine columns from encoders
    cat_cols = list(cat_encoders.keys())
    missing_cat = [c for c in cat_cols if c not in X_new.columns]
    if missing_cat:
        raise ValueError(f"X_new is missing categorical columns: {missing_cat}")

    # numeric = everything else
    num_cols = [c for c in X_new.columns if c not in cat_cols]

    # ---- 1) Numeric: float + scale
    X_num = X_new[num_cols].astype(float)
    X_num_scaled = num_scaler.transform(X_num)   # (n, n_num)

    # ---- 2) Categorical: encode each col -> (n, 1) then column-stack -> (n, n_cat)
    cat_arrays = []
    if not Only_scaling:
        for col in cat_cols:
            le = cat_encoders[col]
            s = X_new[col].astype(str).to_numpy()

            # safe transform: map unseen categories to unknown_value
            class_to_id = {cls: i for i, cls in enumerate(le.classes_)}
            enc = np.array([class_to_id.get(v, unknown_value) for v in s], dtype=np.int32).reshape(-1, 1)

            cat_arrays.append(enc)
    else:
        for col in cat_cols:
            enc = X_new[col].to_numpy().reshape(-1, 1)
            # make sure dtype is numeric
            enc = enc.astype(np.int32)
            cat_arrays.append(enc)

    X_cat_enc = np.hstack(cat_arrays) if len(cat_arrays) else np.empty((len(X_new), 0), dtype=np.int32)

    # ---- Final Ridge input (same shape logic as your training code)
    X_enc_scaled = np.hstack([X_num_scaled, X_cat_enc])

    return X_enc_scaled


def prepare_NN_embedding_inputs(
    Xb,
    cat_encoders,
    scaler_X=None,
    feature_names=None,
):
    """
    Prepare multi-input model inputs for an embedding-based MLP or FT-Transformer,
    using feature names to detect categorical vs numeric columns.

    Parameters
    ----------
    Xb : np.ndarray or pd.DataFrame, shape (N, F)
        Data containing both numeric and categorical columns.
    cat_encoders : dict[col_name -> LabelEncoder]
        Dict of category encoders (categorical columns).
    feature_names : list of str or None
        Names of the columns, in the same order as the model expects.
        - If Xb is a DataFrame and feature_names is None, Xb.columns is used.
        - If Xb is an array, feature_names MUST be provided and match Xb.shape[1].

    Returns
    -------
    list of np.ndarray:
        [X_num, cat1_ids, cat2_ids, ...]
        - X_num: (N, n_numeric_features)
        - each cat*_ids: (N, 1) int32, ready for Embedding layers.
    """

    # --- 0) Normalize Xb to array + feature_names ---
    if isinstance(Xb, pd.DataFrame):
        if feature_names is None:
            feature_names = list(Xb.columns)
            X_arr = Xb.to_numpy()
        else:
            # ensure feature_names order is applied
            missing = [c for c in feature_names if c not in Xb.columns]
            if missing:
                raise ValueError(f"The following feature_names are not in DataFrame columns: {missing}")
            X_arr = Xb[feature_names].to_numpy()
    else:
        # assume it's an array-like
        X_arr = np.asarray(Xb)
        if feature_names is None:
            raise ValueError("feature_names must be provided when Xb is not a DataFrame.")
    
    N, F = X_arr.shape
    if len(feature_names) != F:
        raise ValueError(
            f"feature_names length = {len(feature_names)}, but Xb has {F} columns"
        )

    # --- 1) Categorical columns are keys of encoder dict ---
    cat_cols = list(cat_encoders.keys())

    # Map name -> index
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    cat_indices = [name_to_idx[c] for c in cat_cols if c in name_to_idx]

    # Numeric indices = all others
    all_indices = np.arange(F)
    num_indices = np.setdiff1d(all_indices, cat_indices)

    # --- 2) Extract numeric and categorical partitions ---
    X_num = X_arr[:, num_indices]

    # --- 3) (Optional) scaling could go here; for now we keep as-is ---
    if scaler_X!=None:
        X_num_scaled = scaler_X.transform(X_num)
    else:
        X_num_scaled=X_num
    

    # --- 4) Build categorical (N,1) arrays for embedding inputs ---
    # IMPORTANT: categorical values must already be integer-encoded
    cat_inputs = []
    for col_name in cat_cols:
        if col_name not in name_to_idx:
            raise KeyError(
                f"Categorical column '{col_name}' not found in feature_names"
            )
        col_idx = name_to_idx[col_name]
        col_values = X_arr[:, col_idx]
        cat_inputs.append(col_values.astype("int32").reshape(-1, 1))

    # --- 5) Final multi-input tensor list ---
    return [X_num_scaled] + cat_inputs

def postprocess_class_predictions(predictions, target_cols, encoder_dict, mode=None):
    
    target_col = target_cols[0] 
    if mode=='Neural_Net':
        class_belonging=np.argmax(predictions, axis=1)
        df_pred = pd.DataFrame({target_col: class_belonging})
    elif mode=='LIME':
        class_belonging=np.argmax(predictions)
        df_pred = pd.DataFrame({target_col: [class_belonging]})
    else:
        class_belonging=predictions
        df_pred = pd.DataFrame({target_col: class_belonging})

    # Decode using your existing helper
    df_pred = decode_target_column(df_pred, target_col, encoder_dict)

    # Convert back to simple list for preds_dict mapping
    preds_at_x_raw = df_pred[target_col].tolist()
    
    return preds_at_x_raw
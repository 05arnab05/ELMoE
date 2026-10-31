# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:09:17 2025

@author: arnab
"""

import numpy as np
from itertools import product, islice
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Sequence, Tuple, Optional
from typing import Callable, Any
from numbers import Number
import pandas as pd
from itertools import product
import shap

from Cat_variable_functions import decode_target_column, prepare_NN_embedding_inputs, postprocess_class_predictions, transform_cat_encoder, Ridge_prepare_inputs_from_df
from Trained_ML_model_functions import Load_ModelFile, Load_Scalers, get_model_predictions_classification, get_model_predictions_regression
#--------------Callable Wrapper classes-----------
def Lime_wrapper_function(
    model_file_path,
    model_name,
    x_row: pd.Series,
    feature_names: List[str],
    Target_cols:List[str],
    X_background: pd.DataFrame,
    X_cat_encoder=None,
    Y_cat_encoder=None,
    y_index_for_lime: int = 0
):
    """
    LIME wwrapper
    
    Returns the surrogate prediction(s) at x_row (list of floats).
    If Scaler_X is provided, sampling & distances happen in model (scaled) space,
    and model.predict is called on scaled samples.
    """

    # --- parse row and align to feature_names order ---
    model = Load_ModelFile(model_file_path)
    if model_name in ("MLP", "CNN", "Transformer", "Ridge"):
        scaler_X, scaler_Y = Load_Scalers(model_file_path)
    else:
        scaler_X, scaler_Y = None, None
        
        
    #Encode query
    #Encoding query
    if (Y_cat_encoder!=None):
        test_df = x_row.to_frame().T
        test_df_encoded = transform_cat_encoder(test_df, X_cat_encoder)
        x_row=test_df_encoded.iloc[0]
    
    
    parsed = parse_xrow(x_row)  # should return dict or list
    if isinstance(parsed, dict):
        try:
            x_vec = np.array([parsed[f] for f in feature_names], dtype=np.float32)
        except KeyError as e:
            raise ValueError(f"parse_xrow missing feature {e}") from e
    else:
        x_vec = np.asarray(parsed, dtype=np.float32).reshape(-1)
        if x_vec.size != len(feature_names):
            raise ValueError(f"Parsed row has {x_vec.size} values; expected {len(feature_names)}.")

    # --- ensure background matches order & dtype ---
    if not all(f in X_background.columns for f in feature_names):
        missing = [f for f in feature_names if f not in X_background.columns]
        raise ValueError(f"X_background missing columns: {missing}")
    Xb_df = X_background[feature_names]
    Xb_np = Xb_df.to_numpy(dtype=np.float32)

    # --- call robust LIME in raw vs model space depending on scaler ---
    if scaler_X is not None:
        preds, local_feature_attribution, r2 = lime_predict(
            model=model,
            model_name=model_name,
            x_row=x_vec,
            X_background=Xb_np,
            feature_names=feature_names,
            target_cols=Target_cols,
            X_cat_encoder=X_cat_encoder,
            Y_cat_encoder=Y_cat_encoder,
            preprocessor=scaler_X,  # scale raw -> model space
            scaler_Y=scaler_Y,
            predict_fn=lambda Z: model.predict(Z),  # predict on scaled
        )
    else:
        preds, local_feature_attribution, r2 = lime_predict(
            model=model,
            model_name=model_name,
            x_row=x_vec,
            X_background=Xb_np,
            feature_names=feature_names,
            target_cols=Target_cols,
            X_cat_encoder=X_cat_encoder,
            Y_cat_encoder=Y_cat_encoder,
            scaler_Y=scaler_Y,
            preprocessor=None,                              # raw space
            predict_fn=lambda Z: model.predict(Z),
        )

    # (Optional) pick one output if you only need y_index_for_lime
    # return [preds[y_index_for_lime]]
    return preds, local_feature_attribution  # list of length = n_outputs

def Dice_wrapper_function(
    model_file_path,
    model_name,
    x_row: pd.Series,
    Input_features: List[str],
    Target_cols: List[str],
    immutable_features: List,
    output_upgrade_list: List[float],       # goal_delta as a list aligned to Target_cols
    feature_delta_grid: Dict[str, List[float]],
    X_background: pd.DataFrame,             # not used here, but kept for symmetry with LIME
    X_cat_encoder=None,
    Y_cat_encoder=None,
    classification_encoder=None,
    Scaler_X=None,
    Scaler_Y=None,
    top_k: int = 1,
    max_combos=100,
    max_changed_features_per_candidate: int = 3,
):
    """
    DiCE wrapper:
        Returns list of candidates for upgraded prediction

    """
    # optional: assert background contains inputs (parity with LIME)
    missing_b = [f for f in Input_features if f not in X_background.columns]
    if missing_b:
        raise ValueError(f"X_background missing columns: {missing_b}")

    candidates = dice_generate(
        model_file_path=model_file_path,
        model_name=model_name,
        x_row=x_row,
        input_features=Input_features,
        target_cols=Target_cols,
        immutable_features=immutable_features,
        goal_delta=output_upgrade_list,                    # list -> used directly
        feature_delta_grid=feature_delta_grid,
        X_cat_encoder=X_cat_encoder,
        Y_cat_encoder=Y_cat_encoder,
        top_k=top_k,
        max_combos=max_combos,
    )
    CF_row=candidates[0]['x_new']
    if Y_cat_encoder==None:
        CF_row_dict={fname: float(val) for fname, val in zip(Input_features, CF_row)}
    else:
        CF_row_dict={fname: val for fname, val in zip(Input_features, CF_row)}


    
    return CF_row_dict


# ---------- 1) LIME-style local surrogate (no external lime dependency)

def lime_predict(model,
                 model_name,
                 x_row: Sequence[float],
                 X_background: np.ndarray,
                 feature_names:List[str],
                 target_cols:List[str],
                 X_cat_encoder=None,
                 Y_cat_encoder=None,
                 num_samples: int = 4000,
                 kernel_width: float = 0.6,
                 random_state: Optional[int] = 0,
                 use_knn: bool = True,
                 knn_k: int = 200,
                 use_covariance: bool = True,
                 clip_quantiles: Tuple[float, float] = (0.01, 0.99),
                 ridge_alpha: float = 1.0,
                 preprocessor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 predict_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 debug: bool = False,
                 scaler_Y: Optional[Any] = None,          # <<< NEW: target scaler (e.g., StandardScaler for Y)
                 return_scaled: bool = False,             # <<< NEW: also return scaled intercepts if you want
             ) -> Tuple[List[float], List[float], Optional[List[float]]]:
    
    """
    Returns (preds_at_x_raw, local_r2, preds_at_x_scaled_or_None).

    - If scaler_Y is provided, we inverse_transform the surrogate intercepts into raw units.
    - If scaler_Y is None, we assume model outputs are already in raw units.
    """
    rng = np.random.default_rng(random_state)

    # ----- raw copies -----
    x_raw = np.asarray(x_row, dtype=np.float32).reshape(1, -1)
    Xb_raw = np.asarray(X_background, dtype=np.float32)
    assert Xb_raw.ndim == 2 and x_raw.shape[1] == Xb_raw.shape[1]

    # ----- choose space for X -----
    if preprocessor is not None:
        if Y_cat_encoder==None:
            Xb = preprocessor.transform(Xb_raw)
            x  = preprocessor.transform(x_raw)
        elif model_name in ('Transformer', 'MLP'):
            Xb = apply_preprocessor_array_num_cols_only(preprocessor, X_cat_encoder, Xb_raw, feature_names)
            x  = apply_preprocessor_array_num_cols_only(preprocessor, X_cat_encoder, x_raw, feature_names)
        elif model_name in ('Ridge'):
            Xb = Ridge_prepare_inputs_from_df(Xb_raw, preprocessor, X_cat_encoder, feature_names=feature_names,Only_scaling=True)
            x  = Ridge_prepare_inputs_from_df(x_raw, preprocessor, X_cat_encoder, feature_names=feature_names,Only_scaling=True)
        
    else:
        Xb = Xb_raw
        x  = x_raw

    if hasattr(Xb, "toarray"): Xb = Xb.toarray()
    if hasattr(x, "toarray"):  x  = x.toarray()
    Xb = np.asarray(Xb, dtype=np.float32)
    x  = np.asarray(x,  dtype=np.float32).reshape(1, -1)


    # ----- neighborhood & sampling in chosen space -----
    if use_knn:
        nbrs = NearestNeighbors(n_neighbors=min(knn_k, len(Xb)))
        nbrs.fit(Xb)
        _, idx = nbrs.kneighbors(x, return_distance=True)
        Xloc = Xb[idx[0]]
    else:
        Xloc = Xb

    Qlo = np.quantile(Xloc, clip_quantiles[0], axis=0)
    Qhi = np.quantile(Xloc, clip_quantiles[1], axis=0)
    med = np.median(Xloc, axis=0)
    mad = np.median(np.abs(Xloc - med), axis=0)
    sigma = np.where(mad > 1e-12, 1.4826 * mad, Xloc.std(axis=0, ddof=1) + 1e-9).astype(np.float32)
    sigma = np.maximum(sigma, 1e-6)

    if use_covariance:
        C = np.cov(Xloc, rowvar=False) + 1e-6 * np.eye(Xloc.shape[1], dtype=np.float32)
        Z = rng.multivariate_normal(mean=x.flatten(), cov=C, size=num_samples).astype(np.float32)
    else:
        Z = rng.normal(loc=x, scale=sigma, size=(num_samples, Xloc.shape[1])).astype(np.float32)

    Z = np.clip(Z, Qlo, Qhi).astype(np.float32)

    # ----- kernel & weights -----
    Z_std = (Z - x) / (sigma + 1e-12)
    d = np.sqrt((Z_std ** 2).sum(axis=1)).astype(np.float32)
    w = np.exp(-(d ** 2) / (kernel_width ** 2)).astype(np.float32)
    
    #if all weights are ~0, progressively widen the kernel:
    if not np.isfinite(w).all() or float(w.sum()) == 0.0:
        # widen kernel up to a few times to recover weights
        kw = kernel_width
        for _ in range(3):
            kw *= 2.0
            w = np.exp(-(d ** 2) / (kw ** 2)).astype(np.float32)
            if np.isfinite(w).all() and float(w.sum()) > 0.0:
                break

    # final safety: if still degenerate, fall back to uniform weights
    if (not np.isfinite(w).all()) or (float(w.sum()) == 0.0):
        w = np.ones(len(d), dtype=np.float32)

    # optional: normalize weights to sum to 1 (not required by sklearn, but stable)
    wsum = float(w.sum())
    if wsum > 0:
        w = w / wsum
            
    
    
        
    #Prepare input for model
    if (Y_cat_encoder==None):
        Z_input=Z
    elif model_name in ('Transformer', 'MLP'):
        #rounding cat columns in Z
        Z_input=Z_prep_cat_num(Z, X_cat_encoder, feature_names)
        Z_input= prepare_NN_embedding_inputs(Z_input, cat_encoders=X_cat_encoder, feature_names=feature_names)
    elif model_name=='Ridge':
        Z_input=Z_prep_cat_num(Z, X_cat_encoder, feature_names)
        Z_input=Ridge_prepare_inputs_from_df(Z_input, preprocessor, X_cat_encoder, feature_names=feature_names, Only_scaling=True)     
    else:
        Z_input=Z


    # ----- predict (in model space) -----
    Y_scaled = np.asarray(model.predict(Z_input, verbose=0) if predict_fn is None else predict_fn(Z_input))
    
    if Y_scaled.ndim == 1: Y_scaled = Y_scaled.reshape(-1, 1)
    M = Y_scaled.shape[1]

    # ----- fit local ridge on scaled outputs; collect scaled intercepts -----
    preds_at_x_scaled: List[float] = []
    local_r2: List[float] = []
    weights_per_target = []
    for j in range(M):
        
        y = Y_scaled[:, j]
        
        #Fit a Ridge regression model on clouds of data generated near our point of interest
        reg = Ridge(alpha=ridge_alpha, fit_intercept=True)
        reg.fit(Z_std, y, sample_weight=w)
        
        #predict at the pint we need
        preds_at_x_scaled.append(float(reg.intercept_))  # surrogate @ x in *scaled* Y
        yhat = reg.predict(Z_std)
        
        #metrics to get r2
        wy = np.sum(w * y) / (np.sum(w) + 1e-12)
        ss_res = np.sum(w * (y - yhat) ** 2)
        ss_tot = np.sum(w * (y - wy) ** 2) + 1e-12
        local_r2.append(float(1.0 - ss_res / ss_tot))
        
        # --- get weights as dict ---
        coef_dict = {fname: float(beta) for fname, beta in zip(feature_names, reg.coef_)}
        coef_dict["intercept"] = float(reg.intercept_)  # include bias if useful
        weights_per_target.append(coef_dict)
        

    # ----- unscale intercepts into raw Y units if scaler_Y is provided -----
    
        
    if Y_cat_encoder==None and scaler_Y is not None and hasattr(scaler_Y, "inverse_transform"):
        
        preds_arr_scaled = np.array(preds_at_x_scaled, dtype=np.float32).reshape(1, -1)
        preds_at_x_raw = scaler_Y.inverse_transform(preds_arr_scaled).reshape(-1).tolist()
    else:
        preds_at_x_raw = preds_at_x_scaled[:]  # assume already raw
        
        if(Y_cat_encoder!=None):
            #For classification special, do argmax and convert to 1d array
            preds_at_x_raw=postprocess_class_predictions(preds_at_x_raw, target_cols, Y_cat_encoder, mode='LIME')

    
    # map predictions to target names
    if(Y_cat_encoder!=None):
        preds_dict = {tname: val for tname, val in zip(target_cols, preds_at_x_raw)}
    else:
        preds_dict = {tname: float(val) for tname, val in zip(target_cols, preds_at_x_raw)}

    # convert list[dict] -> dict[target]->dict[feature->weight]
    weights_dict = {tname: w for tname, w in zip(target_cols, weights_per_target)}


    return preds_dict, weights_dict, local_r2
    
    
    
    


#---------------Dice---------

def dice_generate(model_file_path:str,
                  model_name:str,
                  x_row: pd.Series,
                  input_features,
                  target_cols: List[str],
                  goal_delta: List[float],
                  feature_delta_grid: Dict[str, List[float]],
                  X_cat_encoder=None,
                  Y_cat_encoder=None,
                  immutable_features: List = None,
                  desired_class=None,
                  top_k: int = 5,
                  max_combos=100,
                 ) -> List[Dict[str, Any]]:
    """
    Generate DiCE-style counterfactuals by perturbing features according to feature_delta_grid,
    while keeping `immutable_features` unchanged.
    """
    # Ensure x_row is a named Series
    if not isinstance(x_row, pd.Series):
        raise TypeError("x_row must be a pandas.Series with named index (feature names).")

    # Normalize/validate input_features (handle boolean mask vs names)
    input_features = _normalize_feature_names(input_features, x_row.index)

    # --- Handle immutable features -------------------------------------------
    if immutable_features is None:
        immutable_set = set()
    else:
        # Normalize immutable feature names against x_row index as well
        immutable_norm = _normalize_feature_names(immutable_features, x_row.index)
        immutable_set = set(immutable_norm)

        # Safety check: all immutable features should be part of the model input
        unknown_immutables = [f for f in immutable_set if f not in input_features]
        if unknown_immutables:
            raise KeyError(
                f"immutable_features contains names not in input_features: {unknown_immutables}"
            )

    # Features that are allowed to change
    mutable_features = [f for f in input_features if f not in immutable_set]
    if len(mutable_features) == 0:
        raise ValueError(
            "No mutable features left: all input_features are marked immutable. "
            "At least one feature must be mutable to generate counterfactuals."
        )

    # Order & validate delta grid to match ONLY mutable features
    missing_in_grid = [f for f in mutable_features if f not in feature_delta_grid]
    if missing_in_grid:
        raise KeyError(f"feature_delta_grid missing keys for mutable features: {missing_in_grid}")

    feature_names = list(mutable_features)  # apply deltas only to mutable model inputs
    
    # Get per feature range of deltas
    delta_lists = []
    for f in feature_names:
        rng = feature_delta_grid[f]
        if not isinstance(rng, (list, tuple, np.ndarray)) or len(rng) == 0:
            raise ValueError(f"feature_delta_grid['{f}'] must be a non-empty list/array.")
        # Require numeric deltas
        try:
            rng = [float(v) for v in rng]
        except Exception:
            raise ValueError(f"feature_delta_grid['{f}'] must be numeric; got {feature_delta_grid[f]}")
        delta_lists.append(rng)

    x_df= pd.DataFrame([x_row.values], columns=input_features)
    if Y_cat_encoder!=None:
        
        ML_predictions =  get_model_predictions_classification(X_test=x_df,
                                                                model_name=model_name,
                                                                X_encoders=X_cat_encoder,
                                                                Y_encoders=Y_cat_encoder,
                                                                target_cols=target_cols,
                                                                model_file_path=model_file_path)
        #output is a list
        y0_flat=ML_predictions                                                   

    else:
        
        ML_predictions = get_model_predictions_regression(X_test=x_df,
                                                          model_name=model_name,
                                                          model_file_path=model_file_path)
        y0=ML_predictions.reshape(1,-1)
        y0_flat = y0_flat = y0.flatten()
        
    
    #Final output should be 2d np array , type: np.ndarray shape: (1, n_outputs)

    if len(goal_delta) != len(y0_flat):
        raise ValueError(
            f"goal_delta length {len(goal_delta)} != number of outputs {len(y0_flat)} "
            f"(targets: {target_cols})"
        )
    if Y_cat_encoder==None:
        goal_delta = np.asarray(goal_delta, dtype=np.float32)
    
    

    # Generate all perturbation combos (only over mutable features)
    combos = list(product(*delta_lists))
    print('Number of DiCE combos: ', len(combos))
    print('limiting it to first: ', max_combos)
    total = len(combos) if max_combos is None else min(len(combos), max_combos)
    
    
    if max_combos is not None:
        combos = islice(combos, max_combos)
        
    candidates = []
    count=0
    for combo in combos:
        #iteration number tracker
        count=count+1
        
        print(f"{count}/{total} iterations done")
        
        
        #Length check
        if len(combo) != len(feature_names):
            raise AssertionError("Delta combo length != number of mutable features")

        x_new = x_row.copy()
        delta_dict = {}

        # Apply deltas only to mutable features
        for f, d in zip(feature_names, combo):
            x_new.at[f] = float(x_new.at[f]) + float(d)
            delta_dict[f] = float(d)

        # Optionally, record 0 change for immutable features (for clarity in output)
        for f in immutable_set:
            delta_dict[f] = 0.0
        
        x_new_df= pd.DataFrame([x_new.values], columns=input_features)
        if Y_cat_encoder!=None:
            
            ML_predictions =  get_model_predictions_classification(X_test=x_new_df,
                                                                    model_name=model_name,
                                                                    X_encoders=X_cat_encoder,
                                                                    Y_encoders=Y_cat_encoder,
                                                                    target_cols=target_cols,
                                                                    model_file_path=model_file_path)
            
            y_pred=ML_predictions
                                                                

        else:
            
            ML_predictions = get_model_predictions_regression(X_test=x_new_df,
                                                              model_name=model_name,
                                                              model_file_path=model_file_path)
            
            y_pred=ML_predictions.reshape(1,-1)
            y_pred = y_pred.flatten()
        
        

        if Y_cat_encoder==None:
            # Distance from goal (achieved Δy minus desired Δy)
            goal_dist = float(np.linalg.norm((y_pred - y0_flat) - goal_delta))
        elif Y_cat_encoder!=None:
            goal_dist = 0.0 if y_pred == desired_class else 1e9
            
        

        candidates.append({
            "deltas": delta_dict,                    # includes mutable deltas (+ 0.0 for immutable)
            "x_new": x_new[input_features].tolist(), # full input vector, unchanged immutables
            "y_pred": y_pred.tolist() if Y_cat_encoder is None else y_pred,
            "goal_distance": goal_dist
        })
    
    print('out of the loop')
    candidates_sorted = sorted(candidates, key=lambda c: c["goal_distance"])
    return candidates_sorted[:top_k]
    
                  
                 
                  



    

def Shap_feature_attributions(
    model_name: str,
    model,
    x_row: Sequence[float],
    X_background: Optional[np.ndarray] = None,
    y_index: int = 0,
    l1_normalize: bool = True,
    nsamples: Optional[int] = None,
    predict_fn: Optional[callable] = None,
) -> List[float]:
    """
    Wrapper for SHAP attributions across model types.

    Parameters
    ----------
    model_type : str
        One of {"linear", "ridge", "tree", "forest", "xgb", "lgbm", "cat", "keras", "torch", "generic"}.
    model : object
        Trained model (sklearn, Keras, PyTorch, etc.).
    x_row : Sequence[float]
        Single input row in the same preprocessed feature space used by the model.
    X_background : np.ndarray, optional
        Background dataset for SHAP (required for "linear", "ridge", "keras", "torch", "generic").
    y_index : int, default=0
        Target index for multi-output models.
    l1_normalize : bool, default=True
        Whether to L1-normalize the attribution vector.
    nsamples : int, optional
        Number of samples for KernelExplainer (only used if model_type="keras"/"torch"/"generic").
    predict_fn : callable, optional
        Custom prediction function (for PyTorch or wrapped models).
    """

    
    if model_name in ("linear", "Ridge"):
        if X_background is None:
            raise ValueError("X_background is required for SHAP LinearExplainer.")
        return shap_linear_attributions(model, x_row, X_background,
                                        y_index=y_index, l1_normalize=l1_normalize)

    elif model_name in ("DT", "RF", "xgb", "lgbm", "cat"):
        return shap_tree_attributions(model, x_row,
                                      y_index=y_index, l1_normalize=l1_normalize)

    elif model_name in ("MLP", "Transformer", "generic"):
        if X_background is None:
            raise ValueError("X_background is required for SHAP KernelExplainer / DeepExplainer.")
        return shap_kernel_attributions(model, x_row, X_background,
                                        y_index=y_index, predict_fn=predict_fn,
                                         l1_normalize=l1_normalize)

    else:
        raise ValueError(f"Unsupported model_type: {model_name}")

#----------SHAP ----------------
# ============================================================
# 1) LINEAR / RIDGE — use SHAP LinearExplainer
# ============================================================
def shap_linear_attributions(
    model,                          # sklearn linear/ridge with .predict
    x_row: Sequence[float],
    X_background: np.ndarray,       # same preprocessing/space as model input
    y_index: int = 0,
    l1_normalize: bool = True,
    interventional: bool = True
) -> List[float]:
    """
    SHAP baseline for linear models. Returns a list of length F.
    """
    x = _to_2d_row(x_row)
    X_bg = np.asarray(X_background, dtype=np.float32)
    perturb = "interventional" if interventional else "legacy"
    explainer = shap.LinearExplainer(model, X_bg, feature_perturbation=perturb)

    phi = explainer.shap_values(x)  # linear: (1,F) or list[(1,F)] for multi-output
    if isinstance(phi, list):
        phi_vec = np.asarray(phi[y_index]).reshape(-1)
    else:
        phi_vec = np.asarray(phi).reshape(-1)
    return _to_list(phi_vec, l1_normalize)

# ============================================================
# 2) TREES (DT / RF / GBMs) — use SHAP TreeExplainer
# ============================================================
def shap_tree_attributions(
    model,                          # sklearn tree/forest or XGBoost/LightGBM/CatBoost
    x_row: Sequence[float],
    y_index: int = 0,
    l1_normalize: bool = True,
    interventional: bool = True,
    model_output: str = "raw"       # or "margin" for some GBMs
) -> List[float]:
    """
    SHAP baseline for tree models. Returns a list of length F.
    """
    x = _to_2d_row(x_row)
    perturb = "interventional" if interventional else "tree_path_dependent"
    explainer = shap.TreeExplainer(model, feature_perturbation=perturb, model_output=model_output)

    phi = explainer.shap_values(x)  # tree: (1,F) or list[(1,F)] for multi-output
    if isinstance(phi, list):
        phi_vec = np.asarray(phi[y_index]).reshape(-1)
    else:
        phi_vec = np.asarray(phi).reshape(-1)
    return _to_list(phi_vec, l1_normalize)

# ============================================================
# 3) GENERIC / NEURAL NETS — use SHAP KernelExplainer
# ============================================================
def shap_kernel_attributions(
    model,                          # arbitrary model (Keras/PyTorch/sklearn)
    x_row: Sequence[float],
    X_background: np.ndarray,       # keep small (e.g., 50–200 rows)
    y_index: int = 0,
    predict_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    l1_normalize: bool = True,
    random_seed: int = 0
) -> List[float]:
    """
    Model-agnostic SHAP via KernelExplainer. Returns a list of length F.
    For multi-output, selects column y_index.
    """
    rng = np.random.default_rng(random_seed)
    x = _to_2d_row(x_row)
    X_bg = np.asarray(X_background, dtype=np.float32)

    # deterministic wrapper that returns a 1D array for the selected target
    def f_target(X: np.ndarray) -> np.ndarray:
        if predict_fn is not None:
            Y = predict_fn(X)
        else:
            # Keras .predict may accept verbose kw; try both
            try:
                Y = model.predict(X)
            except TypeError:
                Y = model.predict(X, verbose=0)
        Y = np.asarray(Y)
        if Y.ndim == 1:
            return Y
        return Y[:, y_index]

    explainer = shap.KernelExplainer(f_target, X_bg)
    # nsamples: default None lets SHAP choose; you can pass, e.g., 2*F or 10*F
    phi = explainer.shap_values(x)  # kernel: (F,) or (1,F)
    phi_vec = np.asarray(phi).reshape(-1)
    return _to_list(phi_vec, l1_normalize)



# ---------- Utilities------------------------------------------

def Z_prep_cat_num(Z, classification_encoder, feature_names):
    # classification_encoder is assumed to be dict: col_name -> LabelEncoder
    cat_names = list(classification_encoder.keys())
    name_to_idx = {name: i for i, name in enumerate(feature_names)}

    for col_name in cat_names:
        if col_name not in name_to_idx:
            continue
        j = name_to_idx[col_name]  # column index in Z
        le = classification_encoder[col_name]
        n_cat = len(le.classes_)   # categories are 0..n_cat-1 if you used LabelEncoder

        # Snap: round to nearest int, clip into [0, n_cat-1]
        col_vals = np.rint(Z[:, j]).astype(int)          # round to nearest integer
        col_vals = np.clip(col_vals, 0, n_cat - 1)       # ensure within valid range
        Z[:, j] = col_vals.astype(np.float32)
    return Z



def get_keras_input_dim(model):
    shape = model.input_shape

    # Multi-input model → input_shape is a list/tuple of shapes
    if isinstance(shape, (list, tuple)) and isinstance(shape[0], (list, tuple)):
        shape = shape[-1]   # use last input block

    # Now shape is like: (None, n_features)
    if not isinstance(shape, (list, tuple)):
        raise ValueError(f"Unexpected input_shape format: {shape}")

    return int(shape[-1])


    

def apply_preprocessor_array_num_cols_only(preprocessor, class_encoder, arr, feature_names):
    """
    Apply a numeric-only preprocessor (e.g., StandardScaler) to a NumPy array,
    excluding categorical columns whose *names* are in class_encoder.

    Parameters
    ----------
    preprocessor : fitted scaler (StandardScaler etc.)
    class_encoder : dict or None
        Keys are categorical column *names*.
    arr : np.ndarray, shape (n_samples, n_features)
    feature_names : list[str]
        Column names in the same order as the columns in arr

    Returns
    -------
    np.ndarray : same shape as arr, with numeric columns scaled
    """

    # No preprocessor: do nothing
    if preprocessor is None:
        return arr

    n_features = arr.shape[1]
    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names has length {len(feature_names)} but array has {n_features} columns"
        )

    # No categorical encoder: scale everything
    if class_encoder is None:
        return preprocessor.transform(arr)

    # 1) map name -> index
    name_to_idx = {name: i for i, name in enumerate(feature_names)}

    # 2) get categorical indices from encoder keys
    cat_names = list(class_encoder.keys())
    cat_indices = sorted(name_to_idx[name] for name in cat_names if name in name_to_idx)

    # 3) numeric indices = all others
    all_indices = np.arange(n_features)
    num_indices = np.setdiff1d(all_indices, cat_indices)

    # 4) copy and replace only numeric columns with scaled values
    out = arr.copy()
    if len(num_indices) > 0:
        scaled = preprocessor.transform(arr[:, num_indices])
        out[:, num_indices] = scaled

    return out

def _normalize_feature_names(input_features, x_index) -> List[str]:
    """
    Accepts either a list of names OR a boolean mask aligned to x_index.
    Returns a clean list of column names (strings).
    """
    # If it's a pandas object or ndarray of booleans -> convert mask to names
    if isinstance(input_features, (pd.Series, pd.Index, np.ndarray)) and \
       pd.api.types.is_bool_dtype(input_features):
        mask = np.asarray(input_features, dtype=bool)
        if len(mask) != len(x_index):
            raise ValueError(
                f"Boolean mask length {len(mask)} != index length {len(x_index)}"
            )
        return list(pd.Index(x_index)[mask])

    # If it's a plain sequence: coerce to list
    if isinstance(input_features, (list, tuple, np.ndarray, pd.Index)):
        names = list(input_features)
    else:
        raise TypeError(
            f"input_features must be list-like of names or a boolean mask; got {type(input_features)}"
        )

    # Guard against stray booleans mixed in
    if any(isinstance(f, (bool, np.bool_)) for f in names):
        # If it looks like a mask, convert; otherwise, fail loudly
        if len(names) == len(x_index) and all(isinstance(f, (bool, np.bool_)) for f in names):
            mask = np.asarray(names, dtype=bool)
            names = list(pd.Index(x_index)[mask])
        else:
            raise ValueError("input_features contains booleans; expected column names only.")

    # Validate existence
    missing = [f for f in names if f not in x_index]
    if missing:
        raise ValueError(f"input_features contains names not in x_row: {missing}")

    return names

def _normalize_feature_names(input_features, x_index) -> List[str]:
    """
    Accepts either a list of names OR a boolean mask aligned to x_index.
    Returns a clean list of column names (strings).
    """
    # If it's a pandas object or ndarray of booleans -> convert mask to names
    if isinstance(input_features, (pd.Series, pd.Index, np.ndarray)) and \
       pd.api.types.is_bool_dtype(input_features):
        mask = np.asarray(input_features, dtype=bool)
        if len(mask) != len(x_index):
            raise ValueError(
                f"Boolean mask length {len(mask)} != index length {len(x_index)}"
            )
        return list(pd.Index(x_index)[mask])

    # If it's a plain sequence: coerce to list
    if isinstance(input_features, (list, tuple, np.ndarray, pd.Index)):
        names = list(input_features)
    else:
        raise TypeError(
            f"input_features must be list-like of names or a boolean mask; got {type(input_features)}"
        )

    # Guard against stray booleans mixed in
    if any(isinstance(f, (bool, np.bool_)) for f in names):
        # If it looks like a mask, convert; otherwise, fail loudly
        if len(names) == len(x_index) and all(isinstance(f, (bool, np.bool_)) for f in names):
            mask = np.asarray(names, dtype=bool)
            names = list(pd.Index(x_index)[mask])
        else:
            raise ValueError("input_features contains booleans; expected column names only.")

    # Validate existence
    missing = [f for f in names if f not in x_index]
    if missing:
        raise ValueError(f"input_features contains names not in x_row: {missing}")

    return names

def parse_xrow(x_row_input):
    if isinstance(x_row_input, str):
        return [float(v) for v in x_row_input.split(",")]
    elif isinstance(x_row_input, pd.Series):
        return x_row_input.astype(float).tolist()
    elif isinstance(x_row_input, np.ndarray):
        return x_row_input.astype(float).tolist()
    else:
        raise TypeError(f"Unsupported type for x_row: {type(x_row_input)}")

def _sign_fmt(val: float, digits: int = 2) -> str:
    s = f"{val:+.{digits}f}"
    return s.replace("-", "−")

def _delta_fmt(val: float) -> str:
    if abs(val - int(val)) < 1e-12:
        return _sign_fmt(int(val), 0)
    return _sign_fmt(val, 2)

def _ensure_predict(model_or_predict):
    if callable(model_or_predict) and not hasattr(model_or_predict, "predict"):
        return model_or_predict
    if hasattr(model_or_predict, "predict"):
        return lambda X: model_or_predict.predict(X)
    raise TypeError("Pass a model with .predict(X) or a callable predict(X)->(N,M).")

def _to_2d_row(x_row: Sequence[float]) -> np.ndarray:
    arr = np.asarray(x_row, dtype=np.float32).reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("x_row must be 1D; got shape {}".format(arr.shape))
    return arr

def _l1_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = np.sum(np.abs(vec)) + eps
    return vec / s

def _l1_norm(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = float(np.sum(np.abs(vec))) + eps
    return vec / s

def _to_list(vec: np.ndarray, l1_normalize: bool) -> List[float]:
    v = _l1_norm(vec) if l1_normalize else vec
    return v.astype(np.float32).tolist()


# --- (B) Infer bounds from a DataFrame ---------------------------------------

def infer_feature_bounds_from_df(
    df: pd.DataFrame,
    input_features: List[str],
    mode: str = "minmax",        # "minmax" or "quantile"
    q_low: float = 0.01,
    q_high: float = 0.99,
    pad: Optional[float] = None  # optional padding to widen bounds, e.g. pad=1.0
) -> Dict[str, Tuple[Optional[Number], Optional[Number]]]:
    """
    Build {feature: (lo, hi)} from df for the given input_features.
    - mode="minmax": uses observed min/max
    - mode="quantile": uses quantiles [q_low, q_high]
    - pad (if set) subtracts from lo and adds to hi
    """
    bounds: Dict[str, Tuple[Optional[Number], Optional[Number]]] = {}
    for f in input_features:
        if f not in df.columns:
            continue
        col = df[f].dropna().astype(float)
        if col.empty:
            continue
        if mode == "minmax":
            lo, hi = float(col.min()), float(col.max())
        elif mode == "quantile":
            lo, hi = float(col.quantile(q_low)), float(col.quantile(q_high))
        else:
            raise ValueError("mode must be 'minmax' or 'quantile'")
        if pad is not None:
            lo -= float(pad)
            hi += float(pad)
        bounds[f] = (lo, hi)
    return bounds


def _distance_std(z: np.ndarray, x: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Standardized Euclidean distance row-wise."""
    zs = (z - x) / (sigma + 1e-12)
    return np.sqrt((zs ** 2).sum(axis=1))





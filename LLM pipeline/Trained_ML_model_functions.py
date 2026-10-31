# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 13:15:36 2025

@author: arnab
"""


from joblib import load
from contextlib import contextmanager
import keras.models
import keras.layers
from pathlib import Path
import pandas as pd


from FTransformer_model_classifier import (
    FeaturePositionalEmbedding,
    ContinuousTokenizer,
    PrependCLSToken,
    TakeCLSToken,
)

from Cat_variable_functions import ( 
                                    transform_cat_encoder, prepare_NN_embedding_inputs, 
                                    postprocess_class_predictions, 
                                    Ridge_prepare_inputs_from_df)



def Load_Scalers(model_file):
    p = Path(model_file)
    base = p.stem                      # e.g. "Ridge_model"
    if base.endswith("_model"):
        base = base[:-6]   # drop the '_model'
    X_scaler = load(p.parent / f"{base}_X_scaler.save")
    Y_scaler = load(p.parent / f"{base}_Y_scaler.save")
    return X_scaler, Y_scaler

@contextmanager
def _legacy_take_cls_lambda_patch():
    """
    Older Transformer artifacts saved `take_cls` as a Lambda layer. Keras stores
    Lambda functions as Python bytecode, which is not portable across Python
    versions. Substitute that specific stateless layer during deserialization.
    """
    lambda_classes = {keras.layers.Lambda}
    try:
        from keras.src.layers.core.lambda_layer import Lambda as SrcLambda
        lambda_classes.add(SrcLambda)
    except Exception:
        pass

    originals = {cls: cls.from_config for cls in lambda_classes}

    def patched_from_config(cls, config, *args, **kwargs):
        if isinstance(config, dict) and config.get("name") == "take_cls":
            return TakeCLSToken(
                name=config.get("name", "take_cls"),
                trainable=config.get("trainable", True),
            )
        return originals[cls](config, *args, **kwargs)

    try:
        for cls in lambda_classes:
            cls.from_config = classmethod(patched_from_config)
        yield
    finally:
        for cls, original in originals.items():
            cls.from_config = original

def Load_ModelFile(model_file, compile=True):
        """
        Load a model or object depending on extension.
    
        - .keras, .h5 : Keras model (Functional/Sequential/Model subclass)
        - .joblib     : joblib object (e.g., sklearn model, scaler)
        """
        custom_objs = {
        "FeaturePositionalEmbedding": FeaturePositionalEmbedding,
        "ContinuousTokenizer": ContinuousTokenizer,
        "PrependCLSToken": PrependCLSToken,
        "TakeCLSToken": TakeCLSToken,

        # Some environments look up by the registered name as well:
        "Custom>FeaturePositionalEmbedding": FeaturePositionalEmbedding,
        "Custom>ContinuousTokenizer": ContinuousTokenizer,
        "Custom>PrependCLSToken": PrependCLSToken,
        "Custom>TakeCLSToken": TakeCLSToken,
    }
    
        if model_file.endswith(".keras") or model_file.endswith(".h5"):
            with _legacy_take_cls_lambda_patch():
                model = keras.models.load_model(
                    model_file,
                    custom_objects=custom_objs,
                    compile=compile,
                    safe_mode=False   # <-- allow lambda deserialization
                )
            return model
    
        elif model_file.endswith(".joblib"):
            return load(model_file)
    
        else:
            raise ValueError(f"Unsupported model file format: {model_file}")

def get_model_predictions_classification(
    X_test: pd.DataFrame,
    model_name: str,
    X_encoders: dict,
    Y_encoders: dict,
    target_cols,
    model_file_path: str | None = None,
    model=None,
    compile_model: bool = True,
):
    """
    Generalized classification prediction function (no class / self).

    Parameters
    ----------
    X_test : pd.DataFrame
        Test feature dataframe.
    model_name : str
        Name of the model type, e.g. 'MLP', 'Transformer', 'Ridge', 'RandomForest', etc.
    X_encoders : dict
        Dictionary of encoders used for X (as expected by transform_cat_encoder / prepare_NN_embedding_inputs).
    Y_encoders : dict
        Dictionary of encoders for Y (as expected by postprocess_class_predictions).
    target_cols : list[str] or similar
        Target column names expected by postprocess_class_predictions.
    model_file : str, optional
        Path to the saved model file (.keras/.h5/.joblib). Used if `model` is not provided.
    model : object, optional
        Pre-loaded model instance. If None, it will be loaded from `model_file`.
    compile_model : bool, default True
        Passed to Load_ModelFile for Keras models.

    Returns
    -------
    preds : List
        Decoded predictions as returned by postprocess_class_predictions
    """

    # Load model if not provided
    if model is None:
        if model_file_path is None:
            raise ValueError("Either `model` or `model_file` must be provided.")
        model = Load_ModelFile(model_file_path, compile=compile_model)

    # 1. Encode categories of X
    df_encoded = transform_cat_encoder(X_test, X_encoders)

    # 2. Model-specific preprocessing & prediction
    if model_name in ("MLP", "Transformer"):
        # load scalers
        if model_file_path is None:
            raise ValueError(
                "model_file is required to load scalers for MLP/Transformer."
            )
        scaler_X, scaler_Y = Load_Scalers(model_file_path)

        # Prep Input for NeuralNets
        df_prep = prepare_NN_embedding_inputs(
            df_encoded,
            X_encoders,
            scaler_X=scaler_X
        )

        # predict
        predictions = model.predict(df_prep)

        # postprocess / decode to original label space
        preds = postprocess_class_predictions(
            predictions,
            target_cols,
            Y_encoders,
            mode="Neural_Net",
        )

    elif model_name == "Ridge":
        if model_file_path is None:
            raise ValueError(
                "model_file is required to load scalers for Ridge."
            )
        # load scalers
        scaler_X, scaler_Y = Load_Scalers(model_file_path)

        # NOTE: Ridge branch in your original code uses X_test directly
        df_prep = Ridge_prepare_inputs_from_df(X_test, scaler_X, X_encoders)
        predictions = model.predict(df_prep)

        preds = postprocess_class_predictions(
            predictions,
            target_cols,
            Y_encoders,
        )

    else:
        # For non-neural, non-Ridge models: use encoded features directly
        df_prep = df_encoded
        predictions = model.predict(df_prep)

        preds = postprocess_class_predictions(
            predictions,
            target_cols,
            Y_encoders,
        )

    return preds

def get_model_predictions_regression(
    X_test,
    model_name: str,
    model_file_path: str | None = None,
    model=None,
    compile_model: bool = True,
):
    """
    Generalized regression prediction helper (no class / no self).

    Parameters
    ----------
    X_test : pd.DataFrame or np.ndarray
        Test features.
    model_name : str
        One of: 'MLP', 'Transformer', 'Ridge', 'CNN', or any other sklearn-type model.
    model_file : str, optional
        Path to saved model for loading model + scalers.
    model : object, optional
        Pre-loaded model instance. If None, loaded from model_file.
    compile_model : bool, default True
        Used only when loading Keras models.

    Returns
    -------
    np.ndarray
        Predictions (inverse-transformed if scalers are used).
    """

    # --- Load model if necessary ---
    if model is None:
        if model_file_path is None:
            raise ValueError("Either `model` or `model_file` must be provided.")
        model = Load_ModelFile(model_file_path, compile=compile_model)

    # --- Scaler-based models ---
    if model_name in ("MLP", "Transformer", "Ridge"):
        if model_file_path is None:
            raise ValueError("model_file is required to load scalers for regression models.")
        scaler_X, scaler_Y = Load_Scalers(model_file_path)

        # Scale X and predict
        X_scaled = scaler_X.transform(X_test)
        preds = model.predict(X_scaled)

        # Reverse transform Y
        predictions = scaler_Y.inverse_transform(preds)

    # --- CNN branch ---
    elif model_name == "CNN":
        if model_file_path is None:
            raise ValueError("model_file is required to load scalers for CNN.")
        scaler_X, scaler_Y = Load_Scalers(model_file_path)

        X_scaled = scaler_X.transform(X_test)
        X_cnn = X_scaled.astype("float32").reshape(-1, X_scaled.shape[1], 1)
        preds = model.predict(X_cnn, verbose=0)

        predictions = scaler_Y.inverse_transform(preds)

    # --- Default / sklearn style models ---
    else:
        predictions = model.predict(X_test)

    return predictions

# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 13:15:36 2025

@author: arnab
"""
from FTransformer_model import (
    FeaturePositionalEmbedding,
    ContinuousTokenizer,
    PrependCLSToken,
    TakeCLSToken,
)
from joblib import load
from contextlib import contextmanager
import keras.models
import keras.layers
from pathlib import Path

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

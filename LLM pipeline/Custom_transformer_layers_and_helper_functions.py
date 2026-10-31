# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 14:01:52 2025

@author: arnab
"""

# custom_layers.py  (no training, no prints, no I/O)
# Works on both Keras 3 and tf.keras 2.x
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
try:
    from keras.saving import register_keras_serializable
except Exception:
    try:
        from keras.utils import register_keras_serializable
    except Exception:
        from tensorflow.keras.utils import register_keras_serializable




def _cnn_from_2d(X2d_scaled):
    X = np.asarray(X2d_scaled, dtype=np.float32)
    return X.reshape(X.shape[0], X.shape[1], 1)   # (N, F, 1)

def _tf_from_2d_T1(X2d_scaled):
    X = np.asarray(X2d_scaled, dtype=np.float32)
    return X.reshape(X.shape[0], 1, X.shape[1])   # (N, 1, F)
def ensure_3d_after_scaling( X_arr, scaler_X):
    """
    If X is (N,F) -> scale then reshape to (N,1,F)
    If X is (N,T,F) -> flatten to (N*T,F) to scale per-feature, then reshape back.
    """
    X_arr = np.asarray(X_arr, dtype=float)
    if X_arr.ndim == 2:
        # (N, F)
        X_scaled_2d = scaler_X.transform(X_arr)
        return X_scaled_2d.astype("float32")[:, np.newaxis, :]   # (N, 1, F)
    elif X_arr.ndim == 3:
        # (N, T, F)
        N, T, F = X_arr.shape
        X_flat = X_arr.reshape(-1, F)                           # (N*T, F)
        X_flat_scaled = scaler_X.transform(X_flat)
        return X_flat_scaled.reshape(N, T, F).astype("float32") # (N, T, F)
    else:
        raise ValueError("X must be 2D (N,F) or 3D (N,T,F).")


# Essentially offsets each feature such that they form unique clouds which the NN will use to identify it
@register_keras_serializable(package="Custom")
class FeaturePositionalEmbedding(layers.Layer):
    """
    Learnable embedding over FEATURE axis. Expects (batch, F, d_model).
    Adds E[pos] for pos in [0..F-1] to each token.
    """
    def __init__(self, num_features: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.num_features = int(num_features)
        self.d_model = int(d_model)
        # Create the sublayer here; we'll build it in build()
        self.pos_emb = layers.Embedding(
            input_dim=self.num_features,
            output_dim=self.d_model,
            name="feat_pos_embedding"
        )

    def build(self, input_shape):
        # Ensure the sublayer is built so Keras doesn't warn
        # Embedding doesn't need input_shape, but we must trigger its build.
        self.pos_emb.build((None,))  # dummy shape OK for Embedding
        super().build(input_shape)

    def call(self, x):
        # x: (batch, F, d_model)
        F = tf.shape(x)[1]
        positions = tf.range(F, dtype=tf.int32)          # (F,)
        pos = self.pos_emb(positions)                    # (F, d_model)
        pos = tf.expand_dims(pos, axis=0)                # (1, F, d_model)
        pos = tf.tile(pos, [tf.shape(x)[0], 1, 1])       # (batch, F, d_model)
        return x + pos

    def get_config(self):
        return {**super().get_config(),
                "num_features": self.num_features,
                "d_model": self.d_model}
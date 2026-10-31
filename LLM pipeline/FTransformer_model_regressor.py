# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 15:33:39 2025

@author: arnab
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
# --- Keras serializable helper (for saving/loading) ---
try:
    from keras.saving import register_keras_serializable
except Exception:
    try:
        from keras.utils import register_keras_serializable
    except Exception:
        from tensorflow.keras.utils import register_keras_serializable

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
        self.pos_emb = layers.Embedding(
            input_dim=self.num_features,
            output_dim=self.d_model,
            name="feat_pos_embedding"
        )

    def build(self, input_shape):
        self.pos_emb.build((None,))  # trigger build
        super().build(input_shape)

    def call(self, x):
        # x: (batch, F, d_model)
        F = tf.shape(x)[1]
        positions = tf.range(F, dtype=tf.int32)     # (F,)
        pos = self.pos_emb(positions)               # (F, d_model)
        pos = tf.expand_dims(pos, axis=0)           # (1, F, d_model)
        pos = tf.tile(pos, [tf.shape(x)[0], 1, 1])  # (batch, F, d_model)
        return x + pos

    def get_config(self):
        return {
            **super().get_config(),
            "num_features": self.num_features,
            "d_model": self.d_model
        }

@register_keras_serializable(package="Custom")
class ContinuousTokenizer(layers.Layer):
    """
    Per-feature linear projection: (batch, F) -> (batch, F, d_model),
    with separate weights for each column (faithful to FT-Transformer).
    """
    def __init__(self, num_features: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.num_features = int(num_features)
        self.d_model = int(d_model)
        self.w = self.add_weight(
            shape=(self.num_features, self.d_model),
            initializer="glorot_uniform",
            trainable=True, name="w_proj"
        )
        self.b = self.add_weight(
            shape=(self.num_features, self.d_model),
            initializer="zeros",
            trainable=True, name="b_proj"
        )

    def call(self, x):
        # x: (batch, F)
        # einsum: b f , f d -> b f d
        tokens = tf.einsum("bf,fd->bfd", x, self.w) + self.b
        return tokens

    def get_config(self):
        return {
            **super().get_config(),
            "num_features": self.num_features,
            "d_model": self.d_model
        }

@register_keras_serializable(package="Custom")
class PrependCLSToken(layers.Layer):
    """
    Adds a learnable CLS token to the sequence: (batch, F, d_model) -> (batch, F+1, d_model)
    """
    def __init__(self, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.cls_token = self.add_weight(
            shape=(1, 1, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
            name="cls_token"
        )

    def call(self, x):
        b = tf.shape(x)[0]
        cls = tf.tile(self.cls_token, [b, 1, 1])  # (batch, 1, d_model)
        return tf.concat([cls, x], axis=1)

    def get_config(self):
        return {**super().get_config(), "d_model": self.d_model}

@register_keras_serializable(package="Custom")
class TakeCLSToken(layers.Layer):
    """Extracts the first sequence token without serializing Python bytecode."""

    def call(self, x, mask=None):
        return x[:, 0, :]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# =========================
#  Transformer encoder (Pre-Norm)
# =========================

def transformer_encoder_prenorm(d_model, num_heads, d_ff, dropout, count):
    """
    Pre-Norm Transformer encoder block:
      x -> LN -> MHA -> Drop -> +res -> LN -> FFN -> Drop -> +res
    """
    inp = layers.Input(shape=(None, d_model))  # (batch, T, d_model)

    # PreNorm + MHA
    y = layers.LayerNormalization(epsilon=1e-6, name=f"ln1_{count}")(inp)
    y = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        name=f"mha_{count}"
    )(y, y)
    y = layers.Dropout(dropout, name=f"drop_attn_{count}")(y)
    x = layers.Add(name=f"res_attn_{count}")([inp, y])

    # PreNorm + FFN
    y = layers.LayerNormalization(epsilon=1e-6, name=f"ln2_{count}")(x)
    y = layers.Dense(d_ff, activation="relu", name=f"ff1_{count}")(y)
    y = layers.Dropout(dropout, name=f"drop_ff1_{count}")(y)
    y = layers.Dense(d_model, name=f"ff2_{count}")(y)
    y = layers.Dropout(dropout, name=f"drop_ff2_{count}")(y)
    out = layers.Add(name=f"res_ff_{count}")([x, y])

    return keras.Model(inp, out, name=f"TransformerEncoderBlock{count}")

# =========================
#  Factory
# =========================

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

    # inputs: (batch, F)
    inputs = keras.Input(shape=(num_features,), name="features_scalar", dtype=tf.float32)

    # per-feature tokenization -> (batch, F, d_model)
    x = ContinuousTokenizer(num_features=num_features, d_model=d_model, name="cont_tokenizer")(inputs)

    # add learnable feature embeddings
    x = FeaturePositionalEmbedding(num_features=num_features, d_model=d_model, name="feat_pos_emb")(x)

    # prepend CLS token
    x = PrependCLSToken(d_model=d_model, name="prepend_cls")(x)

    # encoder stack
    for i in range(depth):
        x = transformer_encoder_prenorm(d_model, num_heads, d_ff, dropout, i)(x)

    # final norm, then read from CLS
    x = layers.LayerNormalization(epsilon=1e-6, name="final_ln")(x)
    cls = TakeCLSToken(name="take_cls")(x)

    # head
    cls = layers.Dense(64, activation="relu", name="head_hidden")(cls)
    outputs = layers.Dense(num_outputs, activation="linear", name="pred")(cls)

    model = keras.Model(inputs, outputs, name=name)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mean_squared_error")
    return model

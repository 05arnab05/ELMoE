# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 07:19:29 2025

@author: arnab
"""

from abc import ABC, abstractmethod
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import re
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import ast
import math
import keras.models
from joblib import load
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union


class RAG_template(ABC):
    def __init__(self):
        self.DB = None
        self.training_enabled = False

    @abstractmethod
    def Create_DB(self, Train_dataframe, saved_model="False", path=""):
        pass

    @abstractmethod
    def Retreive(self, query):
        pass


class Baseline_RAG(RAG_template):
    def __init__(self):
        self.DB = None
        self.training_enabled = False

    def Create_DB(self, Train_dataframe, saved_model="False", path=""):
        self.DB = Train_dataframe

    def Retreive(self, query=""):
        sampled_data = self.DB.sample(n=50, random_state=42)
        TD_string = sampled_data.to_string(index=False) + "\n\n"
        return TD_string


class KNN_RAG(RAG_template):
    def __init__(self, metric: str, debug: bool = False):
        self.metric = metric
        self.debug = debug
        self.output_upgrade = None
        self.embedding_DB: Optional[pd.DataFrame] = None
        self.embedding_DB_labels: Optional[List[str]] = None
        self.full_DB: Optional[pd.DataFrame] = None
        self.feature_columns: Optional[List[str]] = None
        self.X: Optional[np.ndarray] = None
        self.nbrs: Optional[NearestNeighbors] = None
        self.scaler: Optional[StandardScaler] = None
        self.training_enabled: bool = False
        # last retrieval for debugging
        self._last: Dict[str, object] = {}

    def Create_DB(
        self,
        Train_dataframe_full: pd.DataFrame,
        embedding_DB_labels: List[str],
        retreived_data_number: int = 30,
        output_upgrade: Optional[
            Union[Dict[str, float], List[float], np.ndarray]
        ] = None,
        saved_model: str = "False",
        path: str = "",
    ):
        print("------Building KNN_RAG Retriever-------")

        self.embedding_DB_labels = list(embedding_DB_labels)
        self.full_DB = Train_dataframe_full
        self.output_upgrade = output_upgrade

        if self.metric == "Counterfactuals":
            # Using the predicted outputs (renamed back to the base names)
            embed = self._resolve_predicted_columns(
                Train_dataframe_full, self.embedding_DB_labels
            )
        else:
            embed = Train_dataframe_full[self.embedding_DB_labels].copy()

        self.embedding_DB = embed
        self.feature_columns = list(embed.columns)

        # scale & fit kNN
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.embedding_DB.values)

        self.nbrs = NearestNeighbors(
            n_neighbors=retreived_data_number, algorithm="auto", metric="euclidean"
        ).fit(self.X)

        if self.debug:
            print("RAG feature columns:", self.feature_columns)
            for c in self.feature_columns:
                v = self.embedding_DB[c].values
                print(f"  {c:>16s}: min={v.min():.3f}  max={v.max():.3f}")

    def Retreive(self, query: str) -> str:
        try:
            original_text = query  # the whole question (which includes "... by +2000")
            if "input data:" in query:
                query = query.split("input data:")[-1].strip().rstrip("?")

            q_vec = self._convert_query_to_vector(query)
            if q_vec is None:
                return (
                    self.full_DB.sample(n=50, random_state=42).to_string(index=False)
                    + "\n\n"
                )

            if self.metric == "Counterfactuals":
                # Extract the explicit per-target upgrades from the question text itself
                # e.g., "increase RSRP by +5, DL_bitrate by +2000"
                inferred = self._parse_target_changes_from_text(original_text)
                if inferred:
                    for feat, delta in inferred.items():
                        if feat in self.embedding_DB_labels:
                            idx = self.embedding_DB_labels.index(feat)
                            q_vec[idx] += float(delta)
                else:
                    q_vec = self._apply_output_upgrade(q_vec, self.output_upgrade)

            q_scaled = self.scaler.transform([q_vec])
            distances, indices = self.nbrs.kneighbors(q_scaled)
            retrieved_df = self.full_DB.iloc[indices[0]]

            # debug info
            self._last = {
                "parsed": dict(zip(self.feature_columns, q_vec)),
                "q_vec": q_vec,
                "q_scaled": q_scaled[0],
                "distances": distances[0],
                "indices": indices[0],
                "original_text": original_text,
                "retrieved_df": retrieved_df.copy(),
            }

            if self.debug:
                print("Parsed (after upgrades):", self._last["parsed"])

            return retrieved_df.to_string(index=False) + "\n\n"

        except Exception as e:
            print("Tabular RAG retrieval failed:", e)
            return self.full_DB.sample(n=50).to_string(index=False) + "\n\n"

    def _convert_query_to_vector(self, query: str) -> Optional[np.ndarray]:
        """
        Extracts values for the features in self.embedding_DB_labels from a free-form query like:
        'RSRQ: -13.0, RSSI: -94.0, Longitude: -8.501135, ... DL_bitrate: 2717.0 is RSRP: -102.35 ...'
        Returns a 1D numpy array ordered as self.feature_columns.
        Missing features -> 0.0 (tweak if you prefer np.nan).
        """

        feats = list(self.embedding_DB_labels or [])
        if not feats:
            return None

        feat_alt = "|".join(re.escape(f) for f in feats)
        num = r"-?\d+(?:\.\d+)?(?:e[+\-]?\d+)?"
        pattern = rf"(?i)\b({feat_alt})\b\s*:\s*({num})"

        matches = re.findall(pattern, query)
        parsed = {k: float(v) for k, v in matches}
        if not parsed:
            return None

        return np.array([parsed.get(f, 0.0) for f in feats], dtype=float)

    def _parse_target_changes_from_text(self, text: str) -> Dict[str, float]:
        """
        Parse fragments like: 'increase RSRP by +5, DL_bitrate by +2000'
        Returns {feature: delta} for features that exist in the index space.
        """
        pairs = re.findall(
            r"(?i)\b([A-Za-z_][A-Za-z0-9_]*)\b\s*by\s*([+\-]?\d+(?:\.\d+)?)",
            text,
        )
        out: Dict[str, float] = {}
        valid = set(self.embedding_DB_labels or [])
        for name, delta in pairs:
            if name in valid:
                out[name] = out.get(name, 0.0) + float(delta)
        return out

    def _apply_output_upgrade(self, vector: np.ndarray, upgrade) -> np.ndarray:
        """
        Apply upgrades to the query vector.
        - list/array same size: element-wise add
        - list/array shorter: add to the *last N* features
        - dict {feature: delta}: add per named feature
        """
        out = vector.copy()
        if upgrade is None:
            return out

        if isinstance(upgrade, dict):
            for feat, delta in upgrade.items():
                if feat in self.embedding_DB_labels:
                    idx = self.embedding_DB_labels.index(feat)
                    out[idx] += float(delta)
            return out

        up = np.asarray(upgrade, dtype=float)
        if up.size == out.size:
            return out + up
        if up.size < out.size:
            out[-up.size :] += up
            return out

        raise ValueError(f"Upgrade length {up.size} exceeds vector length {out.size}.")

    @staticmethod
    def _norm_colname(col: Union[str, tuple]) -> str:
        """
        Normalize a column label to a simple label,
        so \"('DL_bitrate_predicted',)\" matches 'DL_bitrate_predicted'.
        """
        if isinstance(col, tuple):
            col = "_".join(map(str, col))
        s = str(col)
        s = re.sub(r"[^A-Za-z0-9_]+", "", s)
        return s.lower()

    def _resolve_predicted_columns(
        self, df: pd.DataFrame, base_names: List[str]
    ) -> pd.DataFrame:
        """
        For base names like ['CQI','UL_bitrate','DL_bitrate'] find the corresponding
        predicted columns in df:
          - exact: '{name}_predicted'
          - weird printed tuple: \"('name_predicted',)\"
        Returns a DF with those columns, renamed to the base names.
        """
        picked = []
        rename_map = {}

        all_cols = list(df.columns)
        norm_map = {}
        for col in all_cols:
            norm_map[self._norm_colname(col)] = col

        for name in base_names:
            target = f"{name}_predicted"
            if target in df.columns:
                col = target
            else:
                target_norm = self._norm_colname(target)
                col = norm_map.get(target_norm, None)

            if col is None:
                raise KeyError(
                    f"Couldn't locate predicted column for '{name}'. "
                    f"Tried '{target}' and normalized equivalents. "
                    f"Available (first 10): {list(map(str, all_cols))[:10]}"
                )

            picked.append(col)
            rename_map[col] = name

        sub = df[picked].copy()
        sub.rename(columns=rename_map, inplace=True)
        return sub


class VecDB_RAG(RAG_template):

    def __init__(self, metric):
        self.metric = metric
        self.output_upgrade = None
        self.embedding_DB = None
        self.embedding_DB_labels = None
        self.full_DB = None

        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.nbrs = None
        self.training_enabled = False

    def Create_DB(
        self,
        Train_dataframe_full: pd.DataFrame,
        embedding_DB_labels: list,
        output_upgrade: list = None,
        retreived_data_number=30,
        saved_model: str = "False",
        path="",
    ):
        print("------Building Vec_DB Retriever-------")

        if self.metric == "Counterfactuals":
            self.embedding_DB_labels = [
                f"('{item}_predicted',)" for item in embedding_DB_labels
            ]
            self.embedding_DB = Train_dataframe_full[self.embedding_DB_labels]
        else:
            self.embedding_DB_labels = embedding_DB_labels
            self.embedding_DB = Train_dataframe_full[self.embedding_DB_labels]

        self.full_DB = Train_dataframe_full
        self.output_upgrade = output_upgrade

        # Apply filtering on columns used for text encoding
        self.feature_columns = [col for col in self.embedding_DB.columns]

        # Create text representation only from filtered columns
        self.samples_as_text = self.embedding_DB[self.feature_columns].apply(
            lambda row: ", ".join(
                [f"{col}: {row[col]}" for col in self.feature_columns]
            ),
            axis=1,
        )

        # Generate embeddings using pre-trained embedding model
        self.embeddings = self.embedding_model.encode(
            self.samples_as_text.tolist(), show_progress_bar=True
        )

        # Build cosine-similarity-based k-NN index
        self.nbrs = NearestNeighbors(
            n_neighbors=retreived_data_number, algorithm="auto", metric="cosine"
        )
        self.nbrs.fit(self.embeddings)

    def Retreive(self, query: str):
        try:
            # Step 1: Clean input
            if "input data:" in query:
                query = query.split("input data:")[-1].strip().rstrip("?")

            # Step 2: Apply output upgrade for Counterfactuals metric
            if self.metric == "Counterfactuals":
                query = self._apply_output_upgrade_to_text(query, self.output_upgrade)

            # Step 3: Embed query
            query_embedding = self.embedding_model.encode([query])[0]

            # Step 4: Search in embedding space
            distances, indices = self.nbrs.kneighbors([query_embedding])

            # Step 5: Retrieve full rows from full_DB
            retrieved_df = self.full_DB.iloc[indices[0]]

            return retrieved_df.to_string(index=False) + "\n\n"

        except Exception as e:
            print("VecDB_RAG retrieval failed:", e)
            return (
                self.full_DB.sample(n=50, random_state=42).to_string(index=False)
                + "\n\n"
            )

    def _apply_output_upgrade_to_text(
        self, query: str, output_upgrade_CF: list[float]
    ) -> str:
        """
        Parses a query string like "RSRQ: -14.0, RSSI: -85.0, ..." and upgrades the last
        len(output_upgrade_CF) values by the given floats. Returns a modified string.
        """
        try:
            # Extract only valid key: value pairs (e.g., "CQI: 7.0")
            kv_pairs = re.findall(r"([A-Za-z_]+):\s*(-?\d+\.?\d*)", query)

            # Convert to dictionary
            parsed_dict = {key: float(val) for key, val in kv_pairs}

            # Identify the last N features and apply the upgrades
            target_features = list(parsed_dict.keys())[-len(output_upgrade_CF) :]
            for feat, delta in zip(target_features, output_upgrade_CF):
                parsed_dict[feat] += delta

            # Reconstruct updated query string
            upgraded_query = ", ".join(f"{k}: {v}" for k, v in parsed_dict.items())
            return upgraded_query

        except Exception as e:
            print("Output upgrade failed:", e)
            return query  # fallback to original


class HybridVecRAG(RAG_template):
    def __init__(self, metric):
        self.metric = metric
        self.output_upgrade = None
        self.embedding_DB = None
        self.embedding_DB_labels = None
        self.full_DB = None

        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.texts = None
        self.embeddings = None
        self.faiss_index = None
        self.feature_columns = None
        self.nbrs = None
        self.X = None
        self.training_enabled = False

    def Create_DB(
        self,
        Train_dataframe_full: pd.DataFrame,
        embedding_DB_labels: list,
        output_upgrade: list = None,
        retreived_data_number=30,
        saved_model: str = "False",
        path="",
    ):
        print("------Building Hybrid KNN + FAISS Retriever-------")

        if self.metric == "Counterfactuals":
            self.embedding_DB_labels = [
                f"('{item}_predicted',)" for item in embedding_DB_labels
            ]
            self.embedding_DB = Train_dataframe_full[self.embedding_DB_labels]
        else:
            self.embedding_DB_labels = embedding_DB_labels
            self.embedding_DB = Train_dataframe_full[self.embedding_DB_labels]

        self.full_DB = Train_dataframe_full
        self.output_upgrade = output_upgrade

        # ------------------------ #
        # Tabular (KNN) Retrieval
        # ------------------------ #
        self.feature_columns = [col for col in self.embedding_DB.columns]

        self.X = self.embedding_DB[self.feature_columns].values

        self.nbrs = NearestNeighbors(
            n_neighbors=retreived_data_number, algorithm="auto", metric="euclidean"
        )
        self.nbrs.fit(self.X)

        # ------------------------ #
        # Semantic (FAISS) Retrieval
        # ------------------------ #
        self.texts = (
            self.embedding_DB[self.feature_columns]
            .apply(
                lambda row: ", ".join(
                    [f"{col}: {row[col]}" for col in self.feature_columns]
                ),
                axis=1,
            )
            .tolist()
        )

        self.embeddings = self.embedding_model.encode(
            self.texts, show_progress_bar=True
        ).astype("float32")

        dim = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dim)
        self.faiss_index.add(self.embeddings)

    def Retreive(self, query: str, k: int = 30):
        try:
            # Step 1: Clean query
            if "input data:" in query:
                query_clean = query.split("input data:")[-1].strip().rstrip("?")
            else:
                query_clean = query.strip()

            # Step 2: Apply counterfactual output upgrade if needed
            if self.metric == "Counterfactuals":
                query_clean = self._apply_output_upgrade_to_text(
                    query_clean, self.output_upgrade
                )

            # Step 3: Semantic search (FAISS)
            query_embedding = self.embedding_model.encode([query_clean])[0].astype(
                "float32"
            )
            _, faiss_indices = self.faiss_index.search(np.array([query_embedding]), k=k)
            df_faiss = self.full_DB.iloc[faiss_indices[0]]

            # Step 4: Tabular similarity (KNN)
            knn_vector = self._convert_query_to_vector(query_clean)
            if knn_vector is not None:
                _, knn_indices = self.nbrs.kneighbors([knn_vector])
                df_knn = self.full_DB.iloc[knn_indices[0]]
            else:
                df_knn = pd.DataFrame()

            # Step 5: Merge and return unique results
            retrieved_df = pd.concat([df_faiss, df_knn]).drop_duplicates().head(10)
            return retrieved_df.to_string(index=False) + "\n\n"

        except Exception as e:
            print("HybridVecRAG retrieval failed:", e)
            return self.full_DB.sample(n=50).to_string(index=False) + "\n\n"

    def _convert_query_to_vector(self, query: str):
        try:
            kv_pairs = re.findall(r"([A-Za-z_]+):\s*(-?\d+\.?\d*)", query)
            parsed_dict = {key: float(val) for key, val in kv_pairs}
            vector = [parsed_dict.get(feat, 0.0) for feat in self.feature_columns]
            return np.array(vector)
        except Exception as e:
            print("Query parsing failed:", e)
            return None

    def _apply_output_upgrade_to_text(
        self, query: str, output_upgrade_CF: list[float]
    ) -> str:
        """
        Increases the last len(output_upgrade_CF) numeric features in a query string by given deltas.
        Example: "RSRQ: -14.0, RSSI: -85.0, CQI: 9.0, UL_bitrate: 200" with [100, 2000]
        will update UL_bitrate → 300 and the next → +2000.
        """
        try:
            # Parse key-value numeric fields
            kv_pairs = re.findall(r"([A-Za-z_]+):\s*(-?\d+\.?\d*)", query)
            parsed_dict = {k: float(v) for k, v in kv_pairs}

            # Get last N features to apply upgrade
            last_keys = list(parsed_dict.keys())[-len(output_upgrade_CF) :]
            for key, delta in zip(last_keys, output_upgrade_CF):
                parsed_dict[key] += delta

            # Rebuild query string
            upgraded_query = ", ".join(f"{k}: {v}" for k, v in parsed_dict.items())
            return upgraded_query
        except Exception as e:
            print("Output upgrade failed:", e)
            return query


# Everything about this
class Autoencoder_RAG(RAG_template):

    def __init__(self, metric, trainable=True):

        self.metric = metric
        self.output_upgrade = None
        self.embedding_DB = None
        self.embedding_DB_labels = None
        self.full_DB = None

        self.autoencoder = None
        self.embedding_dim = 16
        self.encoder = None
        self.embedding_vectors = None
        self.scaled_query = None
        self.Scaler = None
        self.training_enabled = trainable
        self.LLM_error_history = []  # holds (step, LLM_error) or (index, error)

    def Create_DB(
        self,
        Train_dataframe_full: pd.DataFrame,
        embedding_DB_labels: list,
        output_upgrade: list = None,
        saved_model: str = "False",
        path="",
    ):
        print("------Building ML AutoEncoder Retreiver-------")

        if self.metric == "Counterfactuals":
            self.embedding_DB_labels = [
                f"('{item}_predicted',)" for item in embedding_DB_labels
            ]
            self.embedding_DB = Train_dataframe_full[self.embedding_DB_labels]
        else:
            self.embedding_DB_labels = embedding_DB_labels
            self.embedding_DB = Train_dataframe_full[self.embedding_DB_labels]

        self.full_DB = Train_dataframe_full
        self.output_upgrade = output_upgrade

        self.Scaler = StandardScaler()
        X_scaled = self.Scaler.fit_transform(self.embedding_DB)

        input_dim = self.embedding_DB.shape[1]

        # initialize Autoencoder
        self._build_autoencoder(input_dim, self.embedding_dim)

        # Pre-Train autoencoder
        self.autoencoder.fit(X_scaled, X_scaled, epochs=30, batch_size=32)

        self.embedding_vectors = self.encoder.predict(X_scaled)  # ✅ This now works

    def Retreive(self, query: str, n: int = 50, train_mode: bool = False):
        # Encode everything
        embeddings = self.embedding_vectors

        # Two different transformation based on train or retreive phase
        if train_mode == True:
            np_query = query.values.reshape(1, -1)

        else:
            np_query = self._extract_np_query(query)

        if self.metric == "Counterfactuals":
            np_query = self._apply_output_upgrade(np_query, self.output_upgrade)

        # -----ok till here----
        self.scaled_query = self.Scaler.transform(np_query)
        query_emb = self.encoder.predict(self.scaled_query)

        # Cosine similarity
        similarities = cosine_similarity(embeddings, query_emb).flatten()
        top_k_idx = similarities.argsort()[-n:][::-1]

        # only send embedding DB rows as output while training
        if train_mode == True:
            # top_k_data as Dataframe used for training purposes
            top_k_data = self.embedding_DB.iloc[top_k_idx]

        else:
            # top_k_data as String used directly for output
            top_k_data = self.full_DB.iloc[top_k_idx]
            top_k_data = top_k_data.to_string(index=False) + "\n\n"

        return top_k_data  # Dataframe when called from Train() or string

    def Train(
        self,
        train_data_X: pd.DataFrame,
        train_data_Y: pd.DataFrame,
        RAG_training_parameters: dict,
        epochs: int = 3,
        E_stopping_threshold=0.1,
    ):
        """
        Train the encoder over multiple epochs using LLM error feedback.

        Parameters:
        - epochs: number of full passes over the dataset
        """
        print("------Training RAG-------")

        self.LLM_error_epochwise = []  # Reset or extend

        for epoch in range(epochs):
            epoch_errors = []

            # Shuffle indices
            indices = np.random.permutation(len(train_data_X))

            for j, i in enumerate(indices):
                print(
                    f"\n------ Epoch :{epoch}/{epochs} and  [Sample number: {j}/{len(train_data_X)}] -------"
                )
                # Extract 1 row
                X_row = train_data_X.iloc[i]
                Y_row = train_data_Y.iloc[i]

                # Get LLM error for this sample
                LLM_error = self._get_LLM_error(X_row, Y_row, RAG_training_parameters)

                # Save error to history
                self.LLM_error_history.append((epoch, i, LLM_error))
                epoch_errors.append(LLM_error)

                # Retrieve similar entries and update encoder
                if self.metric == "Accuracy":
                    top_k_data = self.Retreive(X_row, train_mode=True)
                elif self.metric == "Counterfactuals":
                    top_k_data = self.Retreive(Y_row, train_mode=True)

                self._update_model_weights(top_k_data, LLM_error)

            # Track average error for this epoch
            avg_err = np.mean(epoch_errors)
            self.LLM_error_epochwise.append(avg_err)
            print("\n ")
            print(
                f"\n------[Epoch {epoch+1}/{epochs}] Avg LLM Error: {avg_err:.4f}-------"
            )
            print("\n ")
            if (
                epoch > 1
                and abs(self.LLM_error_epochwise[-1] - self.LLM_error_epochwise[-2])
                < E_stopping_threshold
            ):
                print("Stopping early due to convergence.")
                break

        self._plot_LLM_error_history(self.LLM_error_history)

    def _build_autoencoder(
        self,
        input_dim,
        embedding_dim,
    ):
        # Define encoder
        input_layer = layers.Input(shape=(input_dim,))
        x = layers.Dense(64, activation="relu")(input_layer)
        embedding = layers.Dense(embedding_dim)(x)

        # Define decoder
        x = layers.Dense(64, activation="relu")(embedding)
        output_layer = layers.Dense(input_dim)(x)

        self.autoencoder = models.Model(input_layer, output_layer)
        self.encoder = models.Model(input_layer, embedding)

        self.autoencoder.compile(optimizer="adam", loss="mse")

    def _get_LLM_error(
        self,
        train_data_X_row: pd.Series,
        train_data_Y_row: pd.Series,
        RAG_training_parameters: dict,
    ) -> float:

        from Main_modules import (
            Test_data_point_extractor,
            extract_subset_dict,
            Prompt_creator,
            LLM_answers,
        )

        # Convert them to string test datapoints to be used in LLM question
        Train_datapoint_X_string, Train_datapoint_Y_string = Test_data_point_extractor(
            train_data_X_row, train_data_Y_row
        )

        # Create prompt for LLM
        prompt_creator_arguments = extract_subset_dict(
            RAG_training_parameters,
            [
                "metric",
                "ML_model_name",
                "ML_model",
                "meta_data_filename",
                "training_data",
                "output_variable_number",
                "output_upgrade_CF",
                "RAG_object",
            ],
        )
        role, content = Prompt_creator(
            prompt_creator_arguments, Train_datapoint_X_string, Train_datapoint_Y_string
        )

        # Get answers from each LLM for a particular prompt
        LLM_model = RAG_training_parameters["LLM_models"]
        answer_one_datapoint = LLM_answers(LLM_model, role, content)

        if self.metric == "Accuracy":
            Score = self._single_point_accuracy_error(
                RAG_training_parameters["ML_model"],
                train_data_X_row,
                answer_one_datapoint[LLM_model[0]],
                len(RAG_training_parameters["Y_labels"]),
            )
        elif self.metric == "Counterfactuals":
            Score = self._single_point_CF_error(
                RAG_training_parameters["ML_model"],
                answer_one_datapoint[LLM_model[0]],
                Train_datapoint_Y_string,
                len(RAG_training_parameters["X_labels"]),
            )

        return Score

    def _single_point_accuracy_error(
        self,
        model_file: str,
        model_input: pd.Series,
        LLM_answer: dict,
        expected_length: int,
    ) -> float:
        # import model to see model's predictions for test data X
        model = self._load_modelFile(model_file)

        # Reshape model_inputs and convert LLM answers to an array of floats
        model_input = model_input.values.reshape(1, -1)
        LLM_answer_list = ast.literal_eval(LLM_answer[0])

        # Checking length of LLM response
        if len(LLM_answer_list) != expected_length:
            print(
                "\n\n ---------LLM returned wrong amount of arguments. Training example skipped-----"
            )
            return 0

        LLM_answer_float = np.array(LLM_answer_list, dtype=float)

        ground_truth = model.predict(model_input).flatten()
        error = round(np.abs(LLM_answer_float - ground_truth).sum(), 2)

        # Check for Nans and give a disclaimer
        if math.isnan(error):
            print("LLM answer is NaN")
            # replacing NaNs with zero
            error = 0

        return error

    def _single_point_CF_error(
        self,
        model_file: str,
        LLM_answer: dict,
        Train_datapoint_Y_string: str,
        expected_length: int,
    ) -> float:
        # import model to see model's predictions for test data X
        model = self._load_modelFile(model_file)

        LLM_answer_list = ast.literal_eval(LLM_answer[0])

        # Checking length of LLM response
        if len(LLM_answer_list) != expected_length:
            print(
                "\n\n ---------LLM returned wrong amount of arguments. Training example skipped-----"
            )
            return 0

        LLM_answer_float = np.array(LLM_answer_list, dtype=float).reshape(1, -1)

        LLM_CF_prediction = model.predict(LLM_answer_float).flatten()

        np_answer = self._extract_np_query(Train_datapoint_Y_string)
        ground_truth = self._apply_output_upgrade(np_answer, self.output_upgrade)
        error = round(np.abs(LLM_CF_prediction - ground_truth).sum(), 2)

        # Check for Nans and give a disclaimer
        if math.isnan(error):
            print("LLM answer is NaN")
            # replacing NaNs with zero
            error = 0

        return error

    def _load_modelFile(self, model_file):
        if model_file.endswith(".keras"):
            model = keras.models.load_model(model_file)  # Load Keras model
        elif model_file.endswith(".joblib"):
            model = load(model_file)

        return model

    def _update_model_weights(
        self,
        top_k_data: pd.DataFrame,
        LLM_error: float,
        batch_size=16,
        epochs=10,
        learning_rate=1e-2,
    ):
        """
        Fine-tunes an encoder using soft similarity to a query vector.

        Parameters:
        - encoder: trained Keras encoder model
        - query_vector: scaled query vector (shape: (1, feature_dim))
        - top_k_data: scaled top-k similar rows (shape: (k, feature_dim))
        - batch_size: batch size for fine-tuning
        - epochs: number of epochs
        - learning_rate: optimizer learning rate
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        k = top_k_data.shape[0]
        top_k_indices = np.arange(k)
        query_tensor = tf.convert_to_tensor(self.scaled_query, dtype=tf.float32)

        for epoch in range(epochs):
            np.random.shuffle(top_k_indices)
            epoch_loss = 0

            for i in range(0, k, batch_size):
                idx_batch = top_k_indices[i : i + batch_size]
                x_batch = top_k_data.iloc[idx_batch]
                x_tensor = tf.convert_to_tensor(x_batch, dtype=tf.float32)

                loss, grads = self._batch_loss(x_tensor, query_tensor, LLM_error)

                optimizer.apply_gradients(zip(grads, self.encoder.trainable_weights))

                epoch_loss += loss.numpy()

            avg_loss = epoch_loss / (k // batch_size + 1)
            # print(f"[Epoch {epoch + 1}] Avg Loss: {avg_loss:.4f}")

    def _batch_loss(
        self,
        x_tensor,
        query_tensor,
        LLM_error: float,
        beta: float = 0.25,
        w_reg: float = 1e-2,
    ):
        """
        Align cosine similarities with LLM feedback.
        Lower LLM_error → higher similarity target.
        """
        with tf.GradientTape() as tape:
            # ── embeddings ─────────────────────────────────────────────
            emb_batch = self.encoder(x_tensor)
            emb_query = self.encoder(query_tensor)
            emb_query = tf.repeat(emb_query, tf.shape(emb_batch)[0], axis=0)

            # cosine similarity for each retrieved row
            sim_scores = tf.reduce_sum(emb_batch * emb_query, axis=1) / (
                tf.norm(emb_batch, axis=1) * tf.norm(emb_query, axis=1) + 1e-8
            )

            # ── error → similarity target  (exp mapping) ───────────────
            s_target = tf.exp(-beta * LLM_error)  # scalar (0,1]
            s_target = tf.cast(s_target, dtype=tf.float32)
            s_target = tf.fill(tf.shape(sim_scores), s_target)

            # 1) alignment loss
            loss_sim = tf.reduce_mean(tf.square(sim_scores - s_target))

            # 2) small L2 regulariser on embeddings
            loss_reg = tf.reduce_mean(tf.square(emb_batch))

            loss = loss_sim + w_reg * loss_reg

        grads = tape.gradient(loss, self.encoder.trainable_weights)
        return loss, grads

    def _extract_np_query(self, query: str):
        numbers = re.findall(r"-?\d+\.?\d*", query)

        # Step 2: Convert the strings to float
        values = np.array(numbers, dtype=float).reshape(1, -1)  # shape: (1, 6)

        return values

    def _apply_output_upgrade(
        self, np_query: np.ndarray, output_upgrade_CF: list[float]
    ) -> np.ndarray:
        """
        Increases the last len(output_upgrade_CF) values of np_query by the corresponding amounts.

        Parameters:
            np_query (np.ndarray): The input NumPy array of shape (1, N)
            output_upgrade_CF (list): List of float values to be added to the last elements of np_query.

        Returns:
            np.ndarray: Updated array with the upgrades applied.
        """
        # Copy to avoid modifying in-place
        updated = np_query.copy()

        # Apply upgrades to the last len(output_upgrade_CF) elements
        updated[0, -len(output_upgrade_CF) :] += output_upgrade_CF

        return updated

    def _plot_LLM_error_history(
        self, error_history, title="LLM Error per Sample per Epoch"
    ):
        """
        Plot LLM error for each training sample across epochs.

        Parameters:
        - error_history: list of (epoch, sample_index, error_value) tuples
        - title: plot title
        """
        # Convert to DataFrame for easy manipulation
        df = pd.DataFrame(error_history, columns=["epoch", "sample_index", "LLM_error"])

        # Pivot to shape: rows = sample_index, columns = epochs
        pivot_df = df.pivot(index="sample_index", columns="epoch", values="LLM_error")

        # Plot
        plt.figure(figsize=(12, 6))
        for epoch in pivot_df.columns:
            plt.plot(
                pivot_df.index, pivot_df[epoch], label=f"Epoch {epoch+1}", alpha=0.7
            )

        plt.xlabel("Training Sample Index (i)")
        plt.ylabel("LLM Error")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class ML_RL_RAG(RAG_template):
    def __init__(self, trainable=True):
        self.DB = None
        self.training_enabled = trainable

    def Create_DB(self, Train_dataframe, saved_model="False", path=""):
        pass

    def Retreive(self, query):
        pass

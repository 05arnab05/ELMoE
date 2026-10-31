# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 14:35:30 2025

@author: arnab
"""
import pandas as pd
import numpy as np
from typing import List, Iterable, Optional, Union, Sequence, Dict, Any, Tuple
import re
import inspect

from Cat_variable_functions import fit_cat_encoder, transform_cat_encoder, Encode_all

#-------- Similar points Search algorithms-----------

def KNN_similar_points(
        query,
        clustered_df: pd.DataFrame,
        feature_cols: List[str],
        nearby_clusters: Iterable[int],
        X_cat_encoder=None,
        Y_cat_encoder=None,
        k: int = 5,
        cluster_col: str = "Cluster_id",
        use_overlapping: bool = True,           # also include rows whose 'Experts' contains any target cluster
        experts_col: str = "Experts",           # list[int] column produced earlier
        drop_na: bool = True,
        normalize: bool = True,                 # L2-normalize for cosine similarity
        exclude_self: bool = True,              # if query is a row from the same df
        id_cols: Optional[List[str]] = None,    # columns to detect/query self row; if None, fallback to feature match
        debug=False,
        drop_exact_cols=["Cluster_id", "Experts"],
        drop_regex_cols=[r"^P_cluster_\d+$"],
        return_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
    """
    Find the top-k most similar rows (by cosine similarity) to `query`, restricted to
    rows from the specified clusters in `clustered_df`.

    Returns
    -------
    pd.DataFrame
        Top-k rows with columns from the **unencoded** dataframe, plus similarity info
        (which may be dropped at the end if debug=False).
    """

    # Keep an unencoded copy for final output
    clustered_df_unencoded = clustered_df.copy()

    # Optionally encode categorical columns for similarity computation
    if Y_cat_encoder is not None:
        clustered_df = Encode_all(clustered_df_unencoded, X_cat_encoder, Y_cat_encoder)

    # --- Build query vector in the right order ---
    if isinstance(query, (pd.Series, dict)):
        q = np.array([query[c] for c in feature_cols], dtype=float)
    else:
        q = np.asarray(query, dtype=float)
        if q.shape[0] != len(feature_cols):
            raise ValueError("Query length does not match feature_cols length.")

    if np.any(np.isnan(q)):
        raise ValueError("Query contains NaNs in feature columns; set drop_na=False and handle upstream if needed.")

    # --- Filter rows by cluster membership ---
    if isinstance(nearby_clusters, int):
        nearby_clusters = {nearby_clusters}
    else:
        nearby_clusters = set(int(c) for c in nearby_clusters)

    mask = clustered_df[cluster_col].isin(nearby_clusters)
    if use_overlapping and (experts_col in clustered_df.columns):
        # include rows whose Experts list contains any of nearby_clusters
        mask = mask | clustered_df[experts_col].apply(
            lambda lst: bool(set(lst).intersection(nearby_clusters))
            if isinstance(lst, (list, tuple, set)) else False
        )

    sub = clustered_df.loc[mask].copy()
    if sub.empty:
        return pd.DataFrame(columns=["__idx__", cluster_col, "cosine_similarity", "cosine_distance"]
                                      + feature_cols + (return_cols or []))

    # --- Drop NA rows if requested ---
    if drop_na:
        sub = sub.dropna(subset=feature_cols)

    if sub.empty:
        return pd.DataFrame(columns=["__idx__", cluster_col, "cosine_similarity", "cosine_distance"]
                                      + feature_cols + (return_cols or []))

    X = sub[feature_cols].to_numpy(dtype=float)

    # --- Cosine similarity (with optional L2 normalization) ---
    if normalize:
        # L2 normalize rows and query
        def l2norm(A):
            denom = np.linalg.norm(A, axis=1, keepdims=True)
            denom[denom == 0] = 1.0
            return A / denom

        def l2norm_vec(v):
            n = np.linalg.norm(v)
            return v / (n if n != 0 else 1.0)

        Xn = l2norm(X)
        qn = l2norm_vec(q)
        sims = Xn @ qn  # cosine similarity
    else:
        # raw cosine similarity without normalization (less common)
        qn = q
        denom = (np.linalg.norm(X, axis=1) * np.linalg.norm(qn) + 1e-12)
        sims = (X @ qn) / denom

    # --- Exclude the query row itself if it's present in sub ---
    if exclude_self:
        if id_cols and isinstance(query, (pd.Series, dict)):
            # exclude rows with same id_cols values
            cond = np.ones(len(sub), dtype=bool)
            for c in id_cols:
                if c in sub.columns and \
                   ((isinstance(query, dict) and c in query) or
                    (isinstance(query, pd.Series) and c in query.index)):
                    cond &= (sub[c].values != query[c])
            keep_idx = np.where(cond)[0]
        else:
            # fallback: exclude exact feature match (within tolerance)
            tol = 1e-12
            eq_mask = np.all(np.isclose(X, q.reshape(1, -1), atol=tol, rtol=0), axis=1)
            keep_idx = np.where(~eq_mask)[0]

        X = X[keep_idx]
        sims = sims[keep_idx]
        sub = sub.iloc[keep_idx]

    # --- Top-k selection ---
    if len(sims) == 0:
        return pd.DataFrame(columns=["__idx__", cluster_col, "cosine_similarity", "cosine_distance"]
                                      + feature_cols + (return_cols or []))

    top = np.argsort(sims)[::-1][:k]
    sub_top = sub.iloc[top].copy()
    sims_top = sims[top]

    # For the final output, we want the UNENCODED rows corresponding to these indices
    sub_top_unencoded = clustered_df_unencoded.loc[sub_top.index].copy()

    # --- Build output dataframe ---
    result = pd.DataFrame({
        "__idx__": sub_top.index,  # original indices
        cluster_col: sub_top[cluster_col].values,  # cluster id (same in encoded & unencoded)
        "cosine_similarity": sims_top,
        "cosine_distance": 1.0 - sims_top,
    })

    # Attach unencoded columns
    for c in sub_top_unencoded.columns:
        result[c] = sub_top_unencoded[c].values

    # Attach any extra columns explicitly requested (if not already there)
    if return_cols:
        for c in return_cols:
            if c in sub_top_unencoded.columns and c not in result.columns:
                result[c] = sub_top_unencoded[c].values

    # Sort by similarity desc just to be explicit
    result = result.sort_values("cosine_similarity", ascending=False).reset_index(drop=True)

    if not debug:
        # Drop internal columns if requested
        result = result.drop(columns=["__idx__", "cosine_similarity", "cosine_distance"], errors="ignore")
        result = result.drop(
            columns=[c for c in result.columns
                     if c in drop_exact_cols or any(re.match(rx, c) for rx in drop_regex_cols)],
            errors="ignore"
        )
    
    for col in result.columns:
        if pd.api.types.is_numeric_dtype(result[col]):
            result[col] = result[col].round(2)
    

    return result



def Vec_DB_search(dataframe, query, fields):
    #find similar points to query from dataframe
    pass


# =====================================================================
# Additional retrievers (BM25, Dense, Hybrid-RRF, MMR, Cross-encoder)
# =====================================================================

def _filter_by_clusters(clustered_df, nearby_clusters, cluster_col, experts_col, use_overlapping):
    if isinstance(nearby_clusters, int):
        nearby_clusters = {nearby_clusters}
    else:
        nearby_clusters = set(int(c) for c in nearby_clusters)

    mask = clustered_df[cluster_col].isin(nearby_clusters)
    if use_overlapping and (experts_col in clustered_df.columns):
        mask = mask | clustered_df[experts_col].apply(
            lambda lst: bool(set(lst).intersection(nearby_clusters))
            if isinstance(lst, (list, tuple, set)) else False
        )
    return clustered_df.loc[mask].copy()


def _row_to_text(row, feature_cols):
    return ", ".join(f"{c}: {row[c]}" for c in feature_cols)


def _query_to_text(query, feature_cols):
    if isinstance(query, (pd.Series, dict)):
        return ", ".join(f"{c}: {query[c]}" for c in feature_cols)
    q = np.asarray(query, dtype=float).ravel()
    return ", ".join(f"{c}: {v}" for c, v in zip(feature_cols, q))


def _query_to_vec(query, feature_cols):
    if isinstance(query, (pd.Series, dict)):
        return np.array([float(query[c]) for c in feature_cols], dtype=float)
    return np.asarray(query, dtype=float).ravel()


def _build_result_frame(
    sub_unencoded,
    selected_pos,
    scores,
    feature_cols,
    cluster_col,
    drop_exact_cols,
    drop_regex_cols,
    return_cols,
    debug,
):
    sub_top_unencoded = sub_unencoded.iloc[selected_pos].copy()

    result = pd.DataFrame({
        "__idx__": sub_top_unencoded.index,
        cluster_col: sub_top_unencoded[cluster_col].values,
        "cosine_similarity": scores,
        "cosine_distance": 1.0 - np.asarray(scores, dtype=float),
    })

    for c in sub_top_unencoded.columns:
        result[c] = sub_top_unencoded[c].values

    if return_cols:
        for c in return_cols:
            if c in sub_top_unencoded.columns and c not in result.columns:
                result[c] = sub_top_unencoded[c].values

    result = result.sort_values("cosine_similarity", ascending=False).reset_index(drop=True)

    if not debug:
        result = result.drop(columns=["__idx__", "cosine_similarity", "cosine_distance"], errors="ignore")
        result = result.drop(
            columns=[c for c in result.columns
                     if c in drop_exact_cols or any(re.match(rx, c) for rx in drop_regex_cols)],
            errors="ignore"
        )

    for col in result.columns:
        if pd.api.types.is_numeric_dtype(result[col]):
            result[col] = result[col].round(2)

    return result


def _empty_result(feature_cols, cluster_col, return_cols):
    return pd.DataFrame(
        columns=["__idx__", cluster_col, "cosine_similarity", "cosine_distance"]
        + list(feature_cols) + list(return_cols or [])
    )


# ---- BM25 (lexical) --------------------------------------------------

def BM25_similar_points(
    query, clustered_df, feature_cols, nearby_clusters,
    X_cat_encoder=None, Y_cat_encoder=None, k=5,
    cluster_col="Cluster_id", use_overlapping=True, experts_col="Experts",
    drop_na=True, normalize=True, exclude_self=True, id_cols=None, debug=False,
    drop_exact_cols=("Cluster_id", "Experts"), drop_regex_cols=(r"^P_cluster_\d+$",),
    return_cols=None,
):
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as e:
        raise ImportError("BM25 retriever requires `pip install rank_bm25`.") from e

    sub_unencoded = _filter_by_clusters(clustered_df, nearby_clusters, cluster_col, experts_col, use_overlapping)
    if sub_unencoded.empty:
        return _empty_result(feature_cols, cluster_col, return_cols)
    if drop_na:
        sub_unencoded = sub_unencoded.dropna(subset=feature_cols)
    if sub_unencoded.empty:
        return _empty_result(feature_cols, cluster_col, return_cols)

    docs = [_row_to_text(r, feature_cols).split() for _, r in sub_unencoded.iterrows()]
    q_tokens = _query_to_text(query, feature_cols).split()

    bm25 = BM25Okapi(docs)
    scores = np.asarray(bm25.get_scores(q_tokens), dtype=float)

    if exclude_self:
        q_vec = _query_to_vec(query, feature_cols)
        X = sub_unencoded[feature_cols].to_numpy(dtype=float)
        eq = np.all(np.isclose(X, q_vec.reshape(1, -1), atol=1e-12, rtol=0), axis=1)
        scores[eq] = -np.inf

    if normalize and np.max(scores) > 0:
        scores = scores / np.max(scores)

    top = np.argsort(scores)[::-1][:k]
    return _build_result_frame(
        sub_unencoded, top, scores[top], feature_cols, cluster_col,
        drop_exact_cols, drop_regex_cols, return_cols, debug,
    )


# ---- Dense retrieval (sentence-transformers + FAISS) -----------------

_DENSE_CACHE = {}


def _get_dense_model(model_name="all-MiniLM-L6-v2"):
    if model_name in _DENSE_CACHE:
        return _DENSE_CACHE[model_name]
    from sentence_transformers import SentenceTransformer
    _DENSE_CACHE[model_name] = SentenceTransformer(model_name)
    return _DENSE_CACHE[model_name]


def _encode_texts(texts, model_name="all-MiniLM-L6-v2"):
    model = _get_dense_model(model_name)
    emb = model.encode(list(texts), show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(emb, dtype="float32")


def Dense_similar_points(
    query, clustered_df, feature_cols, nearby_clusters,
    X_cat_encoder=None, Y_cat_encoder=None, k=5,
    cluster_col="Cluster_id", use_overlapping=True, experts_col="Experts",
    drop_na=True, normalize=True, exclude_self=True, id_cols=None, debug=False,
    drop_exact_cols=("Cluster_id", "Experts"), drop_regex_cols=(r"^P_cluster_\d+$",),
    return_cols=None, encoder_name="all-MiniLM-L6-v2",
):
    sub_unencoded = _filter_by_clusters(clustered_df, nearby_clusters, cluster_col, experts_col, use_overlapping)
    if sub_unencoded.empty:
        return _empty_result(feature_cols, cluster_col, return_cols)
    if drop_na:
        sub_unencoded = sub_unencoded.dropna(subset=feature_cols)
    if sub_unencoded.empty:
        return _empty_result(feature_cols, cluster_col, return_cols)

    texts = [_row_to_text(r, feature_cols) for _, r in sub_unencoded.iterrows()]
    doc_emb = _encode_texts(texts, encoder_name)
    q_emb = _encode_texts([_query_to_text(query, feature_cols)], encoder_name)[0]

    sims = (doc_emb @ q_emb).astype(float)  # already L2-normalized -> cosine

    if exclude_self:
        q_vec = _query_to_vec(query, feature_cols)
        X = sub_unencoded[feature_cols].to_numpy(dtype=float)
        eq = np.all(np.isclose(X, q_vec.reshape(1, -1), atol=1e-12, rtol=0), axis=1)
        sims[eq] = -np.inf

    top = np.argsort(sims)[::-1][:k]
    return _build_result_frame(
        sub_unencoded, top, sims[top], feature_cols, cluster_col,
        drop_exact_cols, drop_regex_cols, return_cols, debug,
    )


# ---- Hybrid (BM25 + Dense) via Reciprocal Rank Fusion ----------------

def Hybrid_RRF_similar_points(
    query, clustered_df, feature_cols, nearby_clusters,
    X_cat_encoder=None, Y_cat_encoder=None, k=5,
    cluster_col="Cluster_id", use_overlapping=True, experts_col="Experts",
    drop_na=True, normalize=True, exclude_self=True, id_cols=None, debug=False,
    drop_exact_cols=("Cluster_id", "Experts"), drop_regex_cols=(r"^P_cluster_\d+$",),
    return_cols=None, rrf_k=60, candidate_pool=50, encoder_name="all-MiniLM-L6-v2",
):
    """Reciprocal Rank Fusion of BM25 and dense scores.
    score(d) = sum_r 1 / (rrf_k + rank_r(d))
    """
    pool = max(k, candidate_pool)

    bm = BM25_similar_points(
        query, clustered_df, feature_cols, nearby_clusters,
        X_cat_encoder=X_cat_encoder, Y_cat_encoder=Y_cat_encoder, k=pool,
        cluster_col=cluster_col, use_overlapping=use_overlapping, experts_col=experts_col,
        drop_na=drop_na, normalize=False, exclude_self=exclude_self, id_cols=id_cols,
        debug=True,  # need __idx__
        drop_exact_cols=(), drop_regex_cols=(), return_cols=return_cols,
    )
    de = Dense_similar_points(
        query, clustered_df, feature_cols, nearby_clusters,
        X_cat_encoder=X_cat_encoder, Y_cat_encoder=Y_cat_encoder, k=pool,
        cluster_col=cluster_col, use_overlapping=use_overlapping, experts_col=experts_col,
        drop_na=drop_na, normalize=False, exclude_self=exclude_self, id_cols=id_cols,
        debug=True,
        drop_exact_cols=(), drop_regex_cols=(), return_cols=return_cols,
        encoder_name=encoder_name,
    )

    if bm.empty and de.empty:
        return _empty_result(feature_cols, cluster_col, return_cols)

    fused = {}
    for rnk, idx in enumerate(bm["__idx__"].tolist()):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rnk + 1)
    for rnk, idx in enumerate(de["__idx__"].tolist()):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rnk + 1)

    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
    keep_idx = [r[0] for r in ranked]
    scores = np.array([r[1] for r in ranked], dtype=float)
    if normalize and scores.max() > 0:
        scores = scores / scores.max()

    sub_unencoded = clustered_df.loc[keep_idx].copy()
    return _build_result_frame(
        sub_unencoded, list(range(len(sub_unencoded))), scores, feature_cols, cluster_col,
        drop_exact_cols, drop_regex_cols, return_cols, debug,
    )


# ---- MMR (Maximal Marginal Relevance) on top of dense base -----------

def MMR_similar_points(
    query, clustered_df, feature_cols, nearby_clusters,
    X_cat_encoder=None, Y_cat_encoder=None, k=5,
    cluster_col="Cluster_id", use_overlapping=True, experts_col="Experts",
    drop_na=True, normalize=True, exclude_self=True, id_cols=None, debug=False,
    drop_exact_cols=("Cluster_id", "Experts"), drop_regex_cols=(r"^P_cluster_\d+$",),
    return_cols=None, mmr_lambda=0.5, candidate_pool=50, encoder_name="all-MiniLM-L6-v2",
):
    """Diversity-aware reranking via MMR over dense embeddings."""
    sub_unencoded = _filter_by_clusters(clustered_df, nearby_clusters, cluster_col, experts_col, use_overlapping)
    if sub_unencoded.empty:
        return _empty_result(feature_cols, cluster_col, return_cols)
    if drop_na:
        sub_unencoded = sub_unencoded.dropna(subset=feature_cols)
    if sub_unencoded.empty:
        return _empty_result(feature_cols, cluster_col, return_cols)

    texts = [_row_to_text(r, feature_cols) for _, r in sub_unencoded.iterrows()]
    doc_emb = _encode_texts(texts, encoder_name)
    q_emb = _encode_texts([_query_to_text(query, feature_cols)], encoder_name)[0]

    rel = (doc_emb @ q_emb).astype(float)

    if exclude_self:
        q_vec = _query_to_vec(query, feature_cols)
        X = sub_unencoded[feature_cols].to_numpy(dtype=float)
        eq = np.all(np.isclose(X, q_vec.reshape(1, -1), atol=1e-12, rtol=0), axis=1)
        rel[eq] = -np.inf

    pool = min(max(k, candidate_pool), len(rel))
    cand = np.argsort(rel)[::-1][:pool].tolist()

    selected = []
    selected_scores = []
    while cand and len(selected) < k:
        if not selected:
            best = cand[0]
            selected.append(best)
            selected_scores.append(rel[best])
            cand.remove(best)
            continue
        sel_emb = doc_emb[selected]
        cand_emb = doc_emb[cand]
        sim_to_sel = cand_emb @ sel_emb.T  # (n_cand, n_sel)
        max_sim = sim_to_sel.max(axis=1)
        mmr_scores = mmr_lambda * rel[cand] - (1 - mmr_lambda) * max_sim
        best_local = int(np.argmax(mmr_scores))
        best = cand[best_local]
        selected.append(best)
        selected_scores.append(rel[best])
        cand.pop(best_local)

    return _build_result_frame(
        sub_unencoded, selected, np.asarray(selected_scores, dtype=float),
        feature_cols, cluster_col, drop_exact_cols, drop_regex_cols, return_cols, debug,
    )


# ---- Cross-encoder reranker on top of a base retriever ---------------

def CrossEncoder_rerank_similar_points(
    query, clustered_df, feature_cols, nearby_clusters,
    X_cat_encoder=None, Y_cat_encoder=None, k=5,
    cluster_col="Cluster_id", use_overlapping=True, experts_col="Experts",
    drop_na=True, normalize=True, exclude_self=True, id_cols=None, debug=False,
    drop_exact_cols=("Cluster_id", "Experts"), drop_regex_cols=(r"^P_cluster_\d+$",),
    return_cols=None, base_method="dense", candidate_pool=50,
    cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    encoder_name="all-MiniLM-L6-v2",
):
    """Get a candidate pool from a base retriever, then rerank with a cross-encoder."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise ImportError("CrossEncoder reranker requires `pip install sentence-transformers`.") from e

    base_fn = {
        "dense": Dense_similar_points,
        "bm25": BM25_similar_points,
        "knn": KNN_similar_points,
    }.get(base_method.lower())
    if base_fn is None:
        raise ValueError(f"Unknown base_method '{base_method}' for CrossEncoder rerank.")

    base_kwargs = dict(
        query=query, clustered_df=clustered_df, feature_cols=feature_cols,
        nearby_clusters=nearby_clusters, X_cat_encoder=X_cat_encoder, Y_cat_encoder=Y_cat_encoder,
        k=max(k, candidate_pool), cluster_col=cluster_col, use_overlapping=use_overlapping,
        experts_col=experts_col, drop_na=drop_na, normalize=False, exclude_self=exclude_self,
        id_cols=id_cols, debug=True, drop_exact_cols=(), drop_regex_cols=(), return_cols=return_cols,
    )
    if base_method.lower() == "dense":
        base_kwargs["encoder_name"] = encoder_name

    cand = base_fn(**base_kwargs)
    if cand.empty:
        return _empty_result(feature_cols, cluster_col, return_cols)

    sub_unencoded = clustered_df.loc[cand["__idx__"].tolist()].copy()
    q_text = _query_to_text(query, feature_cols)
    pairs = [[q_text, _row_to_text(r, feature_cols)] for _, r in sub_unencoded.iterrows()]

    ce = CrossEncoder(cross_encoder_name)
    scores = np.asarray(ce.predict(pairs), dtype=float)

    top = np.argsort(scores)[::-1][:k]
    if normalize and scores.size and np.max(np.abs(scores)) > 0:
        scores = scores / np.max(np.abs(scores))

    return _build_result_frame(
        sub_unencoded, top, scores[top], feature_cols, cluster_col,
        drop_exact_cols, drop_regex_cols, return_cols, debug,
    )


# ---- Dispatcher ------------------------------------------------------

SIMILAR_POINTS_METHODS = {
    "knn": KNN_similar_points,
    "bm25": BM25_similar_points,
    "dense": Dense_similar_points,
    "hybrid_rrf": Hybrid_RRF_similar_points,
    "mmr": MMR_similar_points,
    "cross_encoder": CrossEncoder_rerank_similar_points,
}


def find_similar_points(method: str, **kwargs):
    """Dispatch to one of the registered similar-points retrievers.

    Parameters
    ----------
    method : str
        One of: 'knn', 'bm25', 'dense', 'hybrid_rrf', 'mmr', 'cross_encoder'.
    **kwargs :
        Forwarded to the selected retriever. See each function's signature.
    """
    key = (method or "knn").lower()
    if key not in SIMILAR_POINTS_METHODS:
        raise ValueError(
            f"Unknown similar-points method '{method}'. "
            f"Choose one of: {list(SIMILAR_POINTS_METHODS)}"
        )
    retriever = SIMILAR_POINTS_METHODS[key]
    signature = inspect.signature(retriever)
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if not accepts_kwargs:
        allowed_kwargs = {
            name
            for name, param in signature.parameters.items()
            if param.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        kwargs = {name: value for name, value in kwargs.items() if name in allowed_kwargs}

    return retriever(**kwargs)


        
        

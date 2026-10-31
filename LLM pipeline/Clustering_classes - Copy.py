# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 16:19:11 2025

@author: arnab
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from typing import List, Iterable, Optional, Union, Sequence, Dict, Any, Tuple
from scipy.stats import norm
import re


#---------------------------------- Main Expert creator class--------------------------------------  
class Clustering_classifier:
    def __init__(self, metric, input_cols, ground_truth_cols, prediction_cols):
        
        self.metric=metric
        self.input_cols=input_cols
        self.ground_truth_cols=ground_truth_cols
        self.prediction_cols=prediction_cols
        
        self.Clusterer_obj=None
        self.Clustered_Df = None
        self.Cluster_keys=None
        self.training_enabled = False
        self.Clustering_cols= None

    def Create_Clusters(self, Train_dataframe, feature_cols, path=""):
        '''Two tasks: Create clusters and cluster keys from Data frame and generate description for each cluster'''
        self.Clustering_cols= feature_cols
        
        self.Clusterer_obj=self._clusterer_object_creator('Gaussian')
        Clustered_dataframe=self.Clusterer_obj.Create_clusters(Train_dataframe, self.Clustering_cols, K_max=5, Prob_cols=True)
        
        self.Clustered_Df=Clustered_dataframe
        
        self.Cluster_text_descriptions= self.Clusterer_obj.Describe_clusters()
        #fix this
        #self.Cluster_characteristics= self._expert_builder(Clustered_dataframe)
        self.Cluster_gradients_descriptions= self._cluster_gradients_Description(Clustered_dataframe)
        
    
    def Retreive_Cluster_expert_answers(self, query, output_upgrade=None):
        '''Retreive string descriptions of clusters similar to query'''
        if(output_upgrade==None):
            Query_list= self._extract_list_of_numbers(query)
        else:
            Query_list= self._extract_and_upgrade_list(query, output_upgrade)
            
        Query_Clusters_dict=self._find_Similar_Clusters(Query_list)
        
        Clusters= [Query_Clusters_dict['Cluster_id']]
        All_cluster_answers=''
        for Cluster_id in Clusters:
            #Cluster_text_description= self.Cluster_text_descriptions[Cluster_id]
        
            #Cluster_gradient_description=self.Cluster_gradients_descriptions[Cluster_id]
            
            Similar_points=self._find_Similar_Points(Query_list, Cluster_id)
            Similar_points_string=f'Cluster {Cluster_id} similar points to query: \n\n'+Similar_points.to_string(index=False).replace("\n", "\n\n")
            
            #Cluster_Expert_answer=Cluster_text_description+ "\n\n"+ Cluster_gradient_description+ "\n\n"+ Similar_points_string+ "\n\n"
            Cluster_Expert_answer= Similar_points_string+ "\n\n"

            All_cluster_answers=All_cluster_answers+Cluster_Expert_answer
        
        return All_cluster_answers

    

    def _clusterer_object_creator(self, Clusterer_name: str):
        
        if (Clusterer_name=='Gaussian'):
            Clusterer_obj=Gaussian_clusterer()
        else:
            print(f'Mistake in name of cluster classifier, please recheck, there is no {Clusterer_name}')
        
        return Clusterer_obj
        
    def _expert_builder(self, clustered_dataframe):
        pass
    
    def _cluster_gradients_Description(self, Clustered_dataframe):
        '''Gradient, and other descriptions of the cluster'''
        effects = self._compute_cluster_gradients(
            Clustered_dataframe,
            feature_cols=self.input_cols,
            target_cols=self.prediction_cols,
            use_soft_core=True,      # use P_cluster_k >= 0.6
            core_threshold=0.6,
            min_core=25,
            add_intercept=True,
            standardize_X=False      # keep “unit change” in original units
        )
        # 3) Format as the desired bullets for, say, Cluster 1:
        bullets = self._format_effects_for_all_clusters(
            effects,
            feature_order=self.input_cols,
            target_order=self.prediction_cols)

        return bullets

    def _extract_list_of_numbers(self, input_str):
        # This regex finds all integers and floats, including negative numbers
        numbers = re.findall(r'-?\d+\.?\d*', input_str)
        # Convert to float for numerical use
        return [float(num) for num in numbers]
    
    
    def _extract_and_upgrade_list(self, input_str, output_upgrade):
        """
        Extract numbers from a string and add corresponding upgrade values.
        
        Parameters
        ----------
        input_str : str
            String containing numbers (e.g. "RSRP -95, DL 2300, SNR 10").
        output_upgrade : list[float]
            List of values to add to extracted numbers (must match length).
        
        Returns
        -------
        list[float]
            Extracted numbers after applying upgrades.
        """
        # Extract all numbers (ints, floats, negatives)
        numbers = re.findall(r'-?\d+\.?\d*', input_str)
        numbers = [float(num) for num in numbers]

        # If lengths don’t match, raise error
        if len(numbers) != len(output_upgrade):
            raise ValueError(
                f"Length mismatch: extracted {len(numbers)} numbers, "
                f"but got {len(output_upgrade)} upgrades."
            )

        # Apply upgrades elementwise
        upgraded = [n + u for n, u in zip(numbers, output_upgrade)]
        return upgraded
    

   
        
    def _find_Similar_Clusters(self,  query):
        '''Search clusters based on key value and query'''
        Clusters=self.Clusterer_obj.Find_experts(query, self.Cluster_keys)
        return Clusters
    
    def _find_Similar_Points(self,  query, Clusters):
        '''Search clusters for similar points to query using functions which index and search similar points'''
       #fix this
        neighbors = KNN_similar_points(
            query,
            self.Clustered_Df,
            feature_cols=self.Clustering_cols,
            nearby_clusters=Clusters,
            k=10,
            use_overlapping=True,            # also include rows whose Experts contains 2 or 5
            return_cols=["Y_true_DL", "Y_pred_DL", "timestamp"]  # optional extra context
        )
        return neighbors
    
    
    def _compute_cluster_gradients(
            self,
            clustered_df: pd.DataFrame,
            feature_cols: List[str],            # X features (inputs only)
            target_cols: List[str],             # Y columns (e.g., predicted outputs or residuals)
            cluster_col: str = "Cluster_id",
            soft_prefix: str = "P_cluster_",
            use_soft_core: bool = True,         # use P_cluster_k >= core_threshold for cluster k
            core_threshold: float = 0.6,
            min_core: int = 25,                 # fallback to hard labels if core too small
            add_intercept: bool = True,         # include intercept in OLS
            standardize_X: bool = False,        # if True, z-score X within cluster before OLS (effects then per-σ)
            drop_na: bool = True,               # drop rows with NA in X or Y
            round_to: int = 3) -> pd.DataFrame:
        """
        For each cluster, fit OLS: Y = X * B + intercept (per target column) on that cluster's core members.
        Returns long-form DataFrame with one row per (cluster, feature, target) containing:
            beta, stderr, t_value, n_used, conf_label
    
        Confidence labels (based on |t|):
            |t| >= 3.0  -> "high conf"
            2.0–3.0     -> "med conf"
            < 2.0       -> "low conf"
    
        Notes:
        - If X'X is singular, uses pseudo-inverse for a stable solution.
        - If standardize_X=True, coefficients are per 1-σ change in the feature (within cluster).
        - Choose target_cols as your *model behavior* proxy:
            * use predicted outputs  -> effects on predictions
            * use residuals (y_true - y_pred) -> effects on errors
        """
        # figure out K from soft columns (if present), else from hard labels
        soft_cols = [c for c in clustered_df.columns if c.startswith(soft_prefix)]
        if soft_cols:
            K = len(soft_cols)
        else:
            K = int(clustered_df[cluster_col].max()) + 1
    
        records: List[Dict] = []
        N_total = len(clustered_df)
    
        for k in range(K):
            # --- select core members for cluster k ---
            if use_soft_core and soft_cols:
                mask = clustered_df[f"{soft_prefix}{k}"] >= core_threshold
                if mask.sum() < min_core:
                    mask = clustered_df[clustered_df[cluster_col] == k].index
                    mask = clustered_df.index.isin(mask)
            else:
                mask = (clustered_df[cluster_col] == k)
    
            subset = clustered_df.loc[mask, feature_cols + target_cols].copy()
            if drop_na:
                subset = subset.dropna(subset=feature_cols + target_cols)
    
            n_used = len(subset)
            if n_used < max(8, len(feature_cols) + 3):
                # too few rows for a reasonable OLS; skip with NaNs
                for f in feature_cols:
                    for y in target_cols:
                        records.append({
                            "cluster": k, "feature": f, "target": y,
                            "beta": np.nan, "stderr": np.nan, "t_value": np.nan,
                            "n_used": n_used, "coverage": round(n_used / max(N_total,1), 4),
                            "conf_label": "insufficient data"
                        })
                continue
    
            # --- build X and Y ---
            X = subset[feature_cols].to_numpy(float)
            if standardize_X:
                mu = X.mean(axis=0, keepdims=True)
                sd = X.std(axis=0, ddof=1, keepdims=True)
                sd_safe = np.where(sd == 0, 1.0, sd)
                X = (X - mu) / sd_safe
                x_scale = sd_safe.flatten()   # for metadata only
            else:
                x_scale = np.ones(X.shape[1])
    
            if add_intercept:
                X_design = np.column_stack([np.ones((X.shape[0], 1)), X])
                feature_names = ["Intercept"] + feature_cols
            else:
                X_design = X
                feature_names = feature_cols
    
            # precompute (X'X)^-1 via pinv for stability
            XtX_inv = np.linalg.pinv(X_design.T @ X_design)
    
            # For each target, fit OLS, compute SE and t
            Y = subset[target_cols].to_numpy(float)  # (n, m)
            # OLS solution for all targets at once:
            beta_mat = XtX_inv @ (X_design.T @ Y)    # shape: (p, m)
            Y_hat = X_design @ beta_mat              # (n, m)
            resid = Y - Y_hat
    
            dof = X_design.shape[0] - X_design.shape[1]
            if dof <= 0:
                sigma2 = np.var(resid, axis=0, ddof=1)  # fallback
            else:
                sigma2 = (resid**2).sum(axis=0) / dof   # (m,)
    
            # standard errors per coefficient per target: sqrt( diag((X'X)^-1) * sigma2_j )
            XtX_diag = np.diag(XtX_inv)                # (p,)
            # Clip tiny negative diagonals to zero to avoid sqrt invalid due to roundoff
            XtX_diag = np.clip(XtX_diag, 0.0, None)
            se_mat = np.sqrt(np.outer(XtX_diag, np.maximum(sigma2, 1e-12)))  # (p, m)
            t_mat = beta_mat / np.where(se_mat == 0, np.nan, se_mat)
    
            # assemble records; skip intercept if present
            start_idx = 1 if add_intercept else 0
            for j_feat, feat in enumerate(feature_names[start_idx:], start=start_idx):
                for j_t, tgt in enumerate(target_cols):
                    beta = float(beta_mat[j_feat, j_t])
                    se = float(se_mat[j_feat, j_t])
                    tval = float(t_mat[j_feat, j_t])
    
                    # confidence labels by |t|
                    if np.isnan(tval):
                        conf = "low conf"
                    else:
                        at = abs(tval)
                        if at >= 3.0:
                            conf = "high conf"
                        elif at >= 2.0:
                            conf = "med conf"
                        else:
                            conf = "low conf"
    
                    records.append({
                        "cluster": k,
                        "feature": feat,
                        "target": tgt,
                        "beta": round(beta, round_to),
                        "stderr": round(se, round_to) if np.isfinite(se) else np.nan,
                        "t_value": round(tval, round_to) if np.isfinite(tval) else np.nan,
                        "n_used": int(n_used),
                        "coverage": round(n_used / max(N_total, 1), 4),
                        "conf_label": conf
                    })
    
        effects_df = pd.DataFrame.from_records(records)
        return effects_df
            
    
    def _format_effects_for_all_clusters(
            self,
            effects_df: pd.DataFrame,
            *,
            feature_order: Optional[List[str]] = None,
            target_order: Optional[List[str]] = None,
            cluster_prefix: str = "Cluster") -> List[str]:
            """
            Produce a list of cluster descriptions (one multiline string per cluster).
            Length of the list equals the number of distinct clusters in effects_df.
    
            Returns
            -------
            list[str]
            """
            clusters = sorted(effects_df["cluster"].dropna().astype(int).unique().tolist())
            out: List[str] = []
            for cid in clusters:
                txt = self._format_effects_for_cluster_text(
                    effects_df,
                    cid,
                    feature_order=feature_order,
                    target_order=target_order,
                    cluster_prefix=cluster_prefix,
                )
                out.append(txt)
            return out
    
    def _format_effects_for_cluster_text(
            self,
            effects_df: pd.DataFrame,
            cluster_id: int,
            *,
            feature_order: Optional[List[str]] = None,
            target_order: Optional[List[str]] = None,
            cluster_prefix: str = "Cluster") -> str:
            """
            Build ONE multiline string for a single cluster, with one line per feature:
              Cluster 1:
                - unit change of SNR: +2200 DL / +0.4 RSRP (high conf)
                - unit change of RSRQ: +900 DL / +0.2 RSRP (med conf)
                ...

            Returns
            -------
            str
            """
            sub = effects_df[effects_df["cluster"] == cluster_id].copy()
            header = f"{cluster_prefix} {cluster_id} gradient description:"
            if sub.empty:
                return f"{header}\n  (insufficient data for effects.)"

            # Optional ordering
            if feature_order is not None:
                sub["__f_ord"] = sub["feature"].map({f: i for i, f in enumerate(feature_order)}).fillna(1e9)
                sub = sub.sort_values(["__f_ord", "feature", "target"])
            else:
                sub = sub.sort_values(["feature", "target"])

            if target_order is not None:
                sub["__t_ord"] = sub["target"].map({t: i for i, t in enumerate(target_order)}).fillna(1e9)
                sub = sub.sort_values(["feature", "__t_ord"])

            # Build one line per feature
            lines = []
            for feat, g in sub.groupby("feature", sort=False):
                parts = []
                for _, r in g.iterrows():
                    sign = "+" if float(r["beta"]) >= 0 else ""
                    parts.append(f"{sign}{r['beta']} {r['target']}")
                # conservative confidence across targets
                conf_rank = {"high conf": 2, "med conf": 1, "low conf": 0}
                conf = min((row["conf_label"] for _, row in g.iterrows()), key=lambda c: conf_rank.get(c, 0))
                lines.append(f"  - unit change of {feat}: " + " / ".join(parts) + f" ({conf})")

            return header + "\n" + "\n".join(lines)
   


 
        
                    
            



#---------------------------------- Clustering Algorithms--------------------------------------    
class Clusterer_template(ABC):
    def __init__(self):
        self.model=None
        self.scaler=None
        self.feature_cols=None

    @abstractmethod
    def Create_clusters(self, Train_dataframe, feature_cols, path=""):
        '''Create clusters and return a dataframe with a new column 'Cluster id'''
        pass

    @abstractmethod
    def Find_experts(self, query):
        '''Find clusters/experts nearest to a given query '''
        pass
    
    @abstractmethod
    def Describe_clusters(self):
        '''Rich english description of each cluster'''
        pass



class Gaussian_clusterer(Clusterer_template):
    def __init__(self):
        self.gmm=None #fitted gmm model
        self.scaler=None #fitted scaler
        self.feature_cols=None
    
    def Create_clusters(
            self,
            df: pd.DataFrame,
            feature_cols: List[str],
            K_max=11,
            *,
            criterion: str = "bic",              # "bic" or "aic" for model selection
            scale: bool = True,                  # standardize features before GMM?
            covariance_type: str = "full",       # "full" | "tied" | "diag" | "spherical"
            reg_covar: float = 1e-5,             # numeric stability for covariances
            n_init: int = 5,                     # GMM restarts (helps avoid bad local minima)
            random_state: int = 42,
        
            # Overlapping-expert assignment controls (soft → selected experts)
            prob_threshold: Optional[float] = None,  # include any cluster with p >= this
            cumprob_threshold: float = 0.8,          # else: take top clusters until cum ≥ this
            max_experts: int = 3,                    # cap number of experts per row
        
            # Output column naming
            Prob_cols=False,
            hard_label_col: str = "Cluster_id",
            soft_prefix: str = "P_cluster_",) -> Tuple[pd.DataFrame, GaussianMixture, Optional[StandardScaler]]:
        
            """
            Fit a Gaussian Mixture Model over selected columns, auto-select K with BIC/AIC,
            and add both hard and soft cluster assignments to the dataframe. Also produce
            an 'Experts' list per row with up to `max_experts` overlapping experts.
    
            Parameters
            ----------
            df : pd.DataFrame
                Source dataframe (unchanged; a copy is returned).
            feature_cols : list[str]
                Column names used as features for clustering (e.g., inputs + error features).
            K_range : range
                Candidate numbers of mixture components to try (e.g., range(1, 9)).
            criterion : {"bic","aic"}
                Information criterion to pick the best K.
            scale : bool
                If True, standardize features (recommended).
            covariance_type : {"full","tied","diag","spherical"}
                GMM covariance structure; "diag" often stabilizes high-d settings.
            reg_covar : float
                Small positive value added to covariances for numerical stability.
            n_init : int
                Number of random initializations of GMM.
            random_state : int
                Seed for reproducibility.
    
            prob_threshold : float or None
                If set, select all clusters with responsibility ≥ threshold. If none pass,
                fall back to cumulative strategy.
            cumprob_threshold : float
                When using cumulative strategy, select top clusters until cumulative
                responsibility ≥ this value.
            max_experts : int
                Maximum experts to keep per row (keeps explanations compact).
    
            hard_label_col : str
                Name of the hard-label column (argmax cluster id).
            soft_prefix : str
                Prefix for soft-probability columns (e.g., "P_cluster_0", ...).
    
            Returns
            -------
            output_df : pd.DataFrame
                Copy of input df with:
                  - hard label column (Cluster_id by default)
                  - soft probability columns (P_cluster_*)
                  - 'Experts' list column (overlapping experts per row)
            """
            # -----------------------------
            # (A) Extract and (optionally) standardize features
            # -----------------------------
            K_range=range(1, K_max)
            self.feature_cols=feature_cols
            X = df[feature_cols].to_numpy()
            scaler = None
            if scale:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
                self.scaler=scaler

            # -----------------------------
            # (B) Model selection: sweep K and pick best by BIC/AIC
            # -----------------------------
            best_score = np.inf
            best_gmm = None

            for k in K_range:
                gmm_k = GaussianMixture(
                    n_components=k,
                    covariance_type=covariance_type,
                    reg_covar=reg_covar,
                    n_init=n_init,
                    random_state=random_state,
                )
                gmm_k.fit(X)
                score = gmm_k.bic(X) if criterion.lower() == "bic" else gmm_k.aic(X)

                if score < best_score:
                    best_score = score
                    best_gmm = gmm_k

            gmm = best_gmm  # chosen mixture
            self.gmm=gmm

            # -----------------------------
            # (C) Soft responsibilities and hard labels
            # -----------------------------
            probs = self.gmm.predict_proba(X)        # N x K responsibilities; row sums to 1
            hard_labels = probs.argmax(axis=1)   # argmax cluster id per row

            # -----------------------------
            # (D) Build output dataframe with hard + soft assignments
            # -----------------------------
            output_df = df.copy()
            output_df[hard_label_col] = hard_labels

            # Add columns: P_cluster_0, P_cluster_1, ..., P_cluster_{K-1}
            if(Prob_cols==True):
                for j in range(probs.shape[1]):
                    output_df[f"{soft_prefix}{j}"] = probs[:, j]
            

            experts_col = []

            for i in range(len(output_df)):
                row_probs = probs[i]                       # shape: (K,)
                ranked = np.argsort(row_probs)[::-1]       # indices sorted by p desc
                selected = []

                if prob_threshold is not None:
                    # Fixed threshold first
                    selected = [int(j) for j, p in enumerate(row_probs) if p >= prob_threshold]

                    # Fallback to cumulative if nothing met the threshold
                    if not selected:
                        cum = 0.0
                        tmp = []
                        for j in ranked:
                            tmp.append(int(j))
                            cum += row_probs[j]
                            if cum >= (cumprob_threshold or 1.0) or len(tmp) >= max_experts:
                                break
                        selected = tmp
                else:
                    # Pure cumulative strategy
                    cum = 0.0
                    tmp = []
                    for j in ranked:
                        tmp.append(int(j))
                        cum += row_probs[j]
                        if cum >= (cumprob_threshold or 1.0) or len(tmp) >= max_experts:
                            break
                    selected = tmp

                # Enforce maximum number of experts (keeps prompts concise)
                selected = selected[:max_experts]

                # Optional: mark as unassigned if using a strict threshold and none pass
                if prob_threshold is not None and len(selected) == 0:
                    selected = [-1]

                experts_col.append(selected)

            output_df["Experts"] = experts_col

            return output_df
    
    def Find_experts(
            self,
            sample,                        # dict | pd.Series | 1D np.array
            feature_cols: List[str],
            prob_threshold: Optional[float] = None,  # e.g. 0.25
            cumprob_threshold: float = 0.8,
            max_experts: int = 3) -> Dict[str, Any]:
        """
        Assign a new sample to GMM experts.

        Returns
        -------
        {
          'Cluster_id': int,             # argmax cluster
          'probs': np.ndarray,           # shape (K,), soft responsibilities
          'Experts': List[int],          # up to max_experts selected via thresholds
        }
        """
        # ---- (A) Build 1xD feature vector in the same column order ----
        if isinstance(sample, (pd.Series, dict)):
            x = np.array([sample[c] for c in feature_cols], dtype=float).reshape(1, -1)
        else:
            # assume it's already a 1D array matching feature_cols order
            x = np.asarray(sample, dtype=float).reshape(1, -1)

        # ---- (B) Apply scaler if used during training ----
        if self.scaler is not None:
            x = self.scaler.transform(x)

        # ---- (C) Soft responsibilities and hard label ----
        probs = self.gmm.predict_proba(x)[0]             # shape (K,)
        hard_label = int(np.argmax(probs))

        # ---- (D) Overlapping expert selection (same logic as training) ----
        ranked = np.argsort(probs)[::-1]
        selected: List[int] = []

        if prob_threshold is not None:
            selected = [int(j) for j, p in enumerate(probs) if p >= prob_threshold]
            if not selected:
                # fallback to cumulative
                cum, tmp = 0.0, []
                for j in ranked:
                    tmp.append(int(j))
                    cum += probs[j]
                    if cum >= (cumprob_threshold or 1.0) or len(tmp) >= max_experts:
                        break
                selected = tmp
        else:
            # pure cumulative
            cum, tmp = 0.0, []
            for j in ranked:
                tmp.append(int(j))
                cum += probs[j]
                if cum >= (cumprob_threshold or 1.0) or len(tmp) >= max_experts:
                    break
            selected = tmp

        selected = selected[:max_experts]

        return {
            "Cluster_id": hard_label,
            "probs": probs,
            "Experts": selected
        }
    
    def Describe_clusters(self):
        cluster_info_df= self._cluster_info()
        Cluster_description_dict= self._rich_text_cluster_descriptions( cluster_info_df)
        return  Cluster_description_dict        
        
        
    def _cluster_info(
        self,
        quantiles: Tuple[float, float] = (0.1, 0.9),
        round_to: int = 3) -> pd.DataFrame:
        """
        Produce per-cluster, per-feature percentile ranges directly from a fitted GMM.
    
        Parameters
        ----------
        gmm : sklearn.mixture.GaussianMixture
            Trained GMM (fitted possibly on scaled features).
        quantiles : (float, float)
            Lower/upper quantiles to report (e.g., (0.1, 0.9) => 10–90%).
        round_to : int
            Decimal places for rounding outputs.
    
        Returns
        -------
        pd.DataFrame with columns:
          ['cluster','feature','q_low','q_high','mean','std']
          All values are in ORIGINAL feature units if scaler is provided.
        """
        K = self.gmm.means_.shape[0]
        D = self.gmm.means_.shape[1]
        assert D == len(self.feature_cols), "feature_cols must match GMM feature dimension"
    
        # Quantile → z-score for standard normal
        ql, qh = quantiles
        zL, zH = norm.ppf([ql, qh])   
    
        # Helper to get per-component variance vector (marginal, per feature)
        def component_variances(k: int) -> np.ndarray:
            ct = self.gmm.covariance_type
            cov = self.gmm.covariances_
            if ct == "full":
                return np.diag(cov[k])
            elif ct == "tied":
                return np.diag(cov)
            elif ct == "diag":
                return cov[k]
            elif ct == "spherical":
                return np.full(D, cov[k])
            else:
                raise ValueError(f"Unsupported covariance_type: {ct}")
    
        rows = []
    
        for k in range(K):
            mu = self.gmm.means_[k].copy()         # in model space (possibly scaled)
            var = component_variances(k).copy()
            std = np.sqrt(np.maximum(var, 1e-12))
    
            # Map back to original units if StandardScaler was used
            if self.scaler is not None:
                # x_orig = x_scaled * scale_ + mean_
                mu = mu * self.scaler.scale_ + self.scaler.mean_
                std = std * self.scaler.scale_
    
            # Build rows per feature
            for j, feat in enumerate(self.feature_cols):
                mean_j = mu[j]
                std_j  = std[j]
                q_low  = mean_j + zL * std_j
                q_high = mean_j + zH * std_j
    
                rows.append({
                    "cluster": k,
                    "feature": feat,
                    "q_low": round(float(q_low), round_to),
                    "q_high": round(float(q_high), round_to),
                    "mean": round(float(mean_j), round_to),
                    "std": round(float(std_j), round_to),
                })
    
        desc_df = pd.DataFrame(rows).sort_values(["cluster", "feature"]).reset_index(drop=True)
        return desc_df
    
    def _rich_text_cluster_descriptions(
            self,
            desc_df: pd.DataFrame,
            cluster_prefix: str = "Cluster",
            threshold: float = 0.1) -> Dict[int, str]:
        """
        Generate natural language descriptions of clusters based on percentile ranges,
        skipping features where q_high - q_low <= threshold.

        Parameters
        ----------
        desc_df : pd.DataFrame
            Output of `describe_clusters_from_gmm()` with columns:
            ["cluster","feature","q_low","q_high",...].
        feature_cols : list[str]
            Features to include in the description (skip errors if desired).
        cluster_prefix : str
            Prefix string before cluster id (default "Cluster").
        threshold : float
            Minimum difference between q_high and q_low to include in description.

        Returns
        -------
        dict: {cluster_id: description_text}
        """
        texts = {}
        # ensure features come out in the given order
        desc_df = desc_df.set_index(["cluster", "feature"]).sort_index()

        for k in sorted(desc_df.index.get_level_values("cluster").unique()):
            parts = []
            for feat in self.feature_cols:
                row = desc_df.loc[(k, feat)]
                lo, hi = row["q_low"], row["q_high"]
                if (hi - lo) > threshold:
                    parts.append(f"{feat} varies from {lo:.2f} to {hi:.2f}")
            if parts:
                body = ", ".join(parts)
            else:
                body = "no strong feature variation"
            text = f"{cluster_prefix} {k} characteristics: {body}."
            texts[int(k)] = text

        return texts
        
    

        

#---------------------------------- Searching Algorithms--------------------------------------    

def KNN_similar_points(
        query,
        clustered_df: pd.DataFrame,
        feature_cols: List[str],
        nearby_clusters: Iterable[int],
        k: int = 5,
        cluster_col: str = "Cluster_id",
        use_overlapping: bool = True,           # also include rows whose 'Experts' contains any target cluster
        experts_col: str = "Experts",           # list[int] column produced earlier
        drop_na: bool = True,
        normalize: bool = True,                  # L2-normalize for cosine similarity
        exclude_self: bool = True,               # if query is a row from the same df
        id_cols: Optional[List[str]] = None, # columns to detect/query self row; if None, fallback to feature match
        debug=False,
        drop_exact_cols=["Cluster_id", "Experts"],
        drop_regex_cols=[r"^P_cluster_\d+$"],
        return_cols: Optional[List[str]] = None  ) -> pd.DataFrame:
        """
        Find the top-k most similar rows (by cosine similarity) to `query`, restricted to
        rows from the specified clusters in `clustered_df`.
    
        Parameters
        ----------
        query : dict | pd.Series | 1D sequence
            Values for `feature_cols` in matching order (if sequence), or mapping (if dict/Series).
        clustered_df : pd.DataFrame
            DataFrame that already contains cluster assignments (and optionally 'Experts').
        feature_cols : list[str]
            Feature columns to compute cosine similarity on.
        nearby_clusters : iterable[int]
            Cluster ids to include (e.g., the top experts/clusters for the query).
        k : int
            Number of neighbors to return.
        cluster_col : str
            Name of the hard cluster id column (default 'Cluster_id').
        use_overlapping : bool
            If True, also include rows where any cluster in `nearby_clusters` appears in `experts_col`.
        experts_col : str
            Column with list of overlapping experts per row (if available).
        drop_na : bool
            Drop rows with NaNs in feature columns (and skip if query has NaNs).
        normalize : bool
            L2-normalize vectors before cosine similarity.
        exclude_self : bool
            Try to remove the exact query row if it exists in the filtered set.
        id_cols : list[str] or None
            If provided, use these columns to detect the same row as `query` for exclusion.
        return_cols : list[str] or None
            Extra columns to include in the output (besides index, cluster, similarity).
    
        Returns
        -------
        pd.DataFrame
            Top-k rows with columns:
              ['__idx__', cluster_col, 'cosine_similarity', 'cosine_distance', <feature_cols>, <return_cols...>]
            Sorted by cosine_similarity (descending).
        """
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
                lambda lst: bool(set(lst).intersection(nearby_clusters)) if isinstance(lst, (list, tuple, set)) else False
            )
    
        sub = clustered_df.loc[mask].copy()
        if sub.empty:
            return pd.DataFrame(columns=["__idx__", cluster_col, "cosine_similarity", "cosine_distance"] + feature_cols + (return_cols or []))
    
        # --- Drop NA rows if requested ---
        if drop_na:
            sub = sub.dropna(subset=feature_cols)
    
        if sub.empty:
            return pd.DataFrame(columns=["__idx__", cluster_col, "cosine_similarity", "cosine_distance"] + feature_cols + (return_cols or []))
    
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
                    if c in sub.columns and ((isinstance(query, dict) and c in query) or (isinstance(query, pd.Series) and c in query.index)):
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
            return pd.DataFrame(columns=["__idx__", cluster_col, "cosine_similarity", "cosine_distance"] + feature_cols + (return_cols or []))
    
        top = np.argsort(sims)[::-1][:k]
        sub_top = sub.iloc[top].copy()
        sims_top = sims[top]
    
        # --- Build output dataframe ---
        out_cols = ["__idx__", cluster_col, "cosine_similarity", "cosine_distance"] + feature_cols
        if return_cols:
            out_cols += [c for c in return_cols if c in sub_top.columns and c not in out_cols]
        
        
            result = pd.DataFrame({
                "__idx__": sub_top.index,  # original indices
                cluster_col: sub_top[cluster_col].values,
                "cosine_similarity": sims_top,
                "cosine_distance": 1.0 - sims_top,
            })
        else:
            result = pd.DataFrame()
            
        
        # Attach features and any extra columns
        for c in clustered_df.columns:
            result[c] = sub_top[c].values
        if return_cols:
            for c in return_cols:
                if c in sub_top.columns:
                    result[c] = sub_top[c].values
    
        # Sort by similarity desc just to be explicit
        result = result.sort_values("cosine_similarity", ascending=False).reset_index(drop=True)
        if not debug:
            result = result.drop(columns=["__idx__", "cosine_similarity", "cosine_distance"], errors="ignore")
            result = result.drop(columns=[c for c in result.columns if c in drop_exact_cols or any(re.match(rx, c) for rx in drop_regex_cols)],errors="ignore")

        return result

def Vec_DB_search(dataframe, query, fields):
    #find similar points to query from dataframe
    pass


        
        

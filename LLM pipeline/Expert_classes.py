# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 12:42:51 2025

@author: arnab
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod
from sklearn.utils import resample

class Expert_template(ABC):
    def __init__(self):
        self.model=None
        self.scaler=None
        self.feature_cols=None

    @abstractmethod
    def Build_expert(self, Clustered_dataframe, feature_cols, path=""):
        '''build expert for each cluster in a dataframe with a new column 'Cluster id' 
            depicting the cluster it belongs to
        '''
        pass

    @abstractmethod
    def Cluster_metrics(self, Clustered_dataframe):
        '''Create useful metrics of each cluster to be fed to the LLM '''
        pass
    
   
    
#-------------SVR expert builder---------------------
class SVR_expert(Expert_template):
    def __init__(
        self,
        feature_cols: List[str],
        target_cols: List[str],
        *,
        kernel: str = "linear",          # gradients supported for 'linear'
        C: float = 1.0,
        epsilon: float = 0.1,
        gamma: str = "scale",            # used if kernel != 'linear'
        standardize: bool = True,
        random_state: int = 42,
        min_cluster_samples: int = 20
    ):
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.standardize = standardize
        self.random_state = random_state
        self.min_cluster_samples = min_cluster_samples

        # models_[cluster][target] -> dict(model, x_scaler, y_scaler, support_index, target_name)
        self.models_: Dict[int, Dict[str, Dict[str, Any]]] = {}
        self.cluster_cols=None

    # ---------- 1) Train one SVR per (cluster, target) ----------
    def Build_expert(
        self,
        df: pd.DataFrame,
        *,
        cluster_col: str = "Cluster_id",
        drop_na: bool = True) -> None:
        
        self.cluster_cols= cluster_col
        self.models_.clear()

        # extract cluster ids   
        cluster_ids = sorted(df[cluster_col].dropna().astype(int).unique().tolist())
        for cid in cluster_ids:
            # cluster_id based subsets of dataframe
            sub = df[df[cluster_col] == cid].copy()
            if drop_na:
                sub = sub.dropna(subset=self.feature_cols + self.target_cols)
            if len(sub) < self.min_cluster_samples:
                # Not enough data to fit a reliable expert
                continue

            self.models_[cid] = {}

            # Fit X scaler (shared within cluster)
            if self.standardize:
                x_scaler = StandardScaler().fit(sub[self.feature_cols].to_numpy(float))
                Xs = x_scaler.transform(sub[self.feature_cols].to_numpy(float))
            else:
                x_scaler = None
                Xs = sub[self.feature_cols].to_numpy(float)

            # Train one SVR per target
            for tgt in self.target_cols:
                y = sub[tgt].to_numpy(float)
                if self.standardize:
                    y_scaler = StandardScaler().fit(y.reshape(-1, 1))
                    ys = y_scaler.transform(y.reshape(-1, 1)).ravel()
                else:
                    y_scaler = None
                    ys = y

                svr = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon, gamma=self.gamma)
                svr.fit(Xs, ys)

                # Map support indices back to original df index
                support_local_idx = svr.support_
                support_global_idx = sub.iloc[support_local_idx].index.to_list()

                self.models_[cid][tgt] = dict(
                    model=svr,
                    x_scaler=x_scaler,
                    y_scaler=y_scaler,
                    support_index=support_global_idx,
                    target_name=tgt   # <-- store target name for bootstrap helper
                )

    # ---------- 2) Gradients + support vectors per cluster ----------
    def Cluster_metrics(
        self,
        df: pd.DataFrame,
        *,
        bootstrap_reps: int = 30,
        min_bootstrap_rows: int = 15,
        remove_negative: bool = False) -> Dict[int, Dict[str, Any]]:
        
        
        if self.kernel != "linear":
            raise NotImplementedError("Gradient extraction is implemented for linear SVR only.")

        results: Dict[int, Dict[str, Any]] = {}
        
        
        for cid, tgt_dict in self.models_.items():
            # Subset cluster rows for bootstrapping and reporting
            sub = df[df[self.cluster_cols] == cid].copy()
            sub = sub.dropna(subset=self.feature_cols + self.target_cols)
            
            # ranked SV tables with alpha (per target + combined)
            important_pts = self._collect_support_rows_with_alpha(df, tgt_dict)

            if len(sub) < max(self.min_cluster_samples, min_bootstrap_rows):
                results[cid] = dict(
                    Gradients=[f"Cluster {cid}: (insufficient data for gradient metrics)"],
                    Important_points=important_pts
                )
                continue

            # Build gradient table across targets
            grad_rows = []
            for tgt, pack in tgt_dict.items():
                w_orig = self._linear_weights_in_original_units(pack)
                se_orig = self._bootstrap_se_for_weights(
                    sub, pack, bootstrap_reps, min_bootstrap_rows
                )
                for j, feat in enumerate(self.feature_cols):
                    grad_rows.append(dict(
                        cluster=cid, feature=feat, target=tgt,
                        beta=float(w_orig[j]),
                        se=(float(se_orig[j]) if se_orig is not None and np.isfinite(se_orig[j]) else np.nan)
                    ))

            grad_df = pd.DataFrame(grad_rows)
            
            #adding confidence label
            def _conf_label(beta, se):
                if not np.isfinite(se) or se == 0:
                    mag = abs(beta)
                    return "high conf" if mag >= 0.75 else ("med conf" if mag >= 0.25 else "low conf")
                tlike = abs(beta) / (se + 1e-12)
                if tlike >= 3.0: return "high conf"
                if tlike >= 2.0: return "med conf"
                return "low conf"

            grad_df["conf_label"] = [_conf_label(b, s) for b, s in zip(grad_df["beta"], grad_df["se"])]
    
            # converting gradients to string
            grad_string= self._gradients_as_string(grad_df, cid, remove_negative=remove_negative)
            
            results[cid] = dict(
                Gradients=grad_string,
                Important_points=important_pts
            )


        return results

    # ---------- helpers ----------

    def _linear_weights_in_original_units(self, pack: Dict[str, Any]) -> np.ndarray:
        """Compute linear SVR weights in original units: w_orig = (w_std * sy) / sx."""
        svr = pack["model"]
        x_scaler: Optional[StandardScaler] = pack["x_scaler"]
        y_scaler: Optional[StandardScaler] = pack["y_scaler"]

        dual = svr.dual_coef_.ravel()      # (#SV,)
        SV = svr.support_vectors_          # in standardized X if scaler used
        w_std = (dual @ SV).ravel()        # (n_features,)

        if x_scaler is not None and y_scaler is not None:
            sx = x_scaler.scale_.astype(float)
            sy = y_scaler.scale_.astype(float)[0]
            w_orig = (w_std * sy) / np.where(sx == 0, 1.0, sx)
        else:
            w_orig = w_std.copy()
        return w_orig

    def _bootstrap_se_for_weights(
        self,
        sub: pd.DataFrame,
        pack: Dict[str, Any],
        bootstrap_reps: int,
        min_bootstrap_rows: int
    ) -> Optional[np.ndarray]:
        """Bootstrap linear SVR to estimate SE of weights (original units)."""
        if bootstrap_reps <= 0:
            return None

        rng = np.random.RandomState(self.random_state)
        x_scaler: Optional[StandardScaler] = pack["x_scaler"]
        y_scaler: Optional[StandardScaler] = pack["y_scaler"]
        target_name: str = pack["target_name"]

        # Precompute scalers' scales if present
        sx = (x_scaler.scale_.astype(float) if x_scaler is not None else None)
        sy = (y_scaler.scale_.astype(float)[0] if y_scaler is not None else 1.0)

        bets = []
        for _ in range(bootstrap_reps):
            bs = resample(sub, replace=True, random_state=rng.randint(0, 10**9))
            if len(bs) < min_bootstrap_rows:
                continue

            Xb = bs[self.feature_cols].to_numpy(float)
            yb = bs[target_name].to_numpy(float).reshape(-1, 1)

            if x_scaler is not None:
                Xb = x_scaler.transform(Xb)
            if y_scaler is not None:
                yb = y_scaler.transform(yb).ravel()

            svr_b = SVR(kernel="linear", C=self.C, epsilon=self.epsilon)
            svr_b.fit(Xb, yb)

            dual_b = svr_b.dual_coef_.ravel()
            SV_b = svr_b.support_vectors_
            w_std_b = (dual_b @ SV_b).ravel()

            if sx is not None and y_scaler is not None:
                w_orig_b = (w_std_b * sy) / np.where(sx == 0, 1.0, sx)
            else:
                w_orig_b = w_std_b
            bets.append(w_orig_b)

        if len(bets) < 5:
            return None
        return np.std(np.vstack(bets), axis=0, ddof=1)

    def _collect_support_rows(self, df: pd.DataFrame, tgt_dict: Dict[str, Any]) -> pd.DataFrame:
        """Union of support-vector rows across targets for a cluster."""
        idx = set()
        for _, pack in tgt_dict.items():
            idx.update(pack["support_index"])
        return df.loc[sorted(idx)].copy() if idx else pd.DataFrame(columns=df.columns)
    
    def _collect_support_rows_with_alpha(
        self,
        df: pd.DataFrame,
        tgt_dict: Dict[str, Dict[str, Any]]) -> Any:
        """
        Build ranked support-vector tables with alpha for each target, and a combined table.
        Returns:
          - if multiple targets: {"per_target": {tgt: df}, "combined": df}
          - if single target: a single DataFrame for that target
        Each DataFrame includes columns: ['alpha','abs_alpha', ...original df cols...],
        sorted by abs_alpha desc.
        """
        per_target: Dict[str, pd.DataFrame] = {}
        # accumulate for combined
        combined_map: Dict[int, float] = {}  # index -> sum of abs alphas across targets

        for tgt, pack in tgt_dict.items():
            
            #Extract model
            svr = pack["model"]
            
            dual = svr.dual_coef_.ravel()            # signed weights (alpha_i - alpha_i*)
            support_local_idx = svr.support_         # indices into the training subset
            support_global_idx = df.iloc[support_local_idx].index               \
                if set(support_local_idx).issubset(df.index) else pack["support_index"]

            # align: dual[i] corresponds to support_global_idx[i]
            rows = df.loc[support_global_idx].copy()
            rows = rows.reset_index().rename(columns={"index":"__orig_index__"})
            rows["alpha"] = dual
            rows["abs_alpha"] = np.abs(dual)
            rows = rows.sort_values("abs_alpha", ascending=False).reset_index(drop=True)

            per_target[tgt] = rows

            # update combined influence (sum abs alpha per original row index)
            for idx, a in zip(support_global_idx, np.abs(dual)):
                combined_map[idx] = combined_map.get(idx, 0.0) + float(a)

        if len(per_target) == 1:
            # single target -> return that one table directly
            return next(iter(per_target.values()))

        # build combined table
        if combined_map:
            comb_idx = sorted(combined_map.keys(), key=lambda i: -combined_map[i])
            combined = df.loc[comb_idx].copy()
            combined = combined.reset_index().rename(columns={"index":"__orig_index__"})
            combined["abs_alpha_sum"] = [combined_map[i] for i in comb_idx]
            combined = combined.sort_values("abs_alpha_sum", ascending=False).reset_index(drop=True)
        else:
            combined = pd.DataFrame(columns=["__orig_index__", "abs_alpha_sum", *df.columns.tolist()])

        return {"per_target": per_target, "combined": combined}
    
    def _gradients_as_string(self, grad_df, cid, remove_negative: bool= False):
        '''Converting dataframe of gradients to String'''
        lines = []
        for feat, g in grad_df.groupby("feature"):
            g2 = g[g["beta"] >= 0] if remove_negative else g
            if g2.empty:
                if remove_negative:
                    continue
                g2 = g
            parts, confs = [], []
            for _, r in g2.iterrows():
                sign = "+" if r["beta"] >= 0 else ""
                parts.append(f"{sign}{round(r['beta'], 3)} increase in {r['target']}")
                confs.append(r["conf_label"])
            rank = {"high conf": 2, "med conf": 1, "low conf": 0}
            conf = min(confs, key=lambda c: rank.get(c, 0)) if confs else "low conf"
            # 🔑 build as one sentence, no dash
            lines.append(f"for +1 increase in {feat} = " + " / ".join(parts) + f" ({conf})")

        if not lines:
            lines = [f"(No effects to report for cluster {cid} with current settings)"]

        # join into one large string with newlines
        grad_string = "\n".join(lines)
        
        return grad_string


#--------------------OLS expert-------------------------------------------------

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

class OLS_expert(Expert_template):
    def __init__(
        self,
        feature_cols: List[str],
        target_cols: List[str],
        *,
        soft_prefix: str = "P_cluster_",
        use_soft_core: bool = True,
        core_threshold: float = 0.6,
        min_core: int = 25,
        add_intercept: bool = True,
        standardize_X: bool = False,     # if True, effects are per 1-σ change
        drop_na: bool = True,
        round_to: int = 3,
        remove_negative: bool = False    # if True, hide negative effects in text
    ):
        self.cluster_col = None
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.soft_prefix = soft_prefix
        self.use_soft_core = use_soft_core
        self.core_threshold = core_threshold
        self.min_core = min_core
        self.add_intercept = add_intercept
        self.standardize_X = standardize_X
        self.drop_na = drop_na
        self.round_to = round_to
        self.remove_negative = remove_negative

        # cached metadata after build_expert
        self._clusters_: List[int] = []

    # ===================== PUBLIC API =====================

    def Build_expert(self, df: pd.DataFrame, cluster_col='Cluster_id') -> None:
        
        self.cluster_col= cluster_col
        """
        Light build step: validates columns and records which clusters exist.
        (OLS itself is fit inside Cluster_metrics per cluster.)
        """
        missing = set(self.feature_cols + self.target_cols + [self.cluster_col]) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in df: {missing}")

        # detect cluster ids
        self._clusters_ = sorted(df[self.cluster_col].dropna().astype(int).unique().tolist())

    def Cluster_metrics(self, df: pd.DataFrame, X_cat_encoder=None) -> Dict[int, Dict[str, Any]]:
        """
        Compute OLS-based gradients per cluster and return:
          { cluster_id: {"Gradients": "<multiline string>"} }
        """

        # Identify categorical columns, if any
        if X_cat_encoder!=None:
            cat_cols = X_cat_encoder.keys()
            numeric_features = [c for c in self.feature_cols if c not in cat_cols]
        else:
            numeric_features=self.feature_cols
        
        if not self._clusters_:
            # allow calling without build_expert first
            self._clusters_ = sorted(df[self.cluster_col].dropna().astype(int).unique().tolist())

        effects_df = self._compute_cluster_gradients(df)
        variability = self.compute_local_variability(df, numeric_features)

        result: Dict[int, Dict[str, Any]] = {}

        for cid in sorted(effects_df["cluster"].dropna().astype(int).unique().tolist()):
            lines = self._format_feature_lines_for_cluster(
                effects_df,
                variability,
                cid,
                numeric_features,      # <-- use numeric-only features
                self.target_cols
            )
            grad_string = "\n".join(lines)
            result[cid] = {"Gradients": grad_string}

        # also include clusters that were empty/insufficient
        for cid in self._clusters_:
            if cid not in result:
                result[cid] = {"Gradients": f"(Cluster {cid}: insufficient data)"}

        return result
    
    

    # ===================== COMPUTE (BROKEN INTO SMALL HELPERS) =====================

    def _compute_cluster_gradients(self, df: pd.DataFrame, cat_variables=None) -> pd.DataFrame:
        """Main dispatcher: loops clusters, fits OLS, aggregates to long-form table."""
        soft_cols = [c for c in df.columns if c.startswith(self.soft_prefix)]
        has_soft = len(soft_cols) > 0

        # Identify categorical columns from the dataframe
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Use only non-categorical features for OLS
        numeric_features = [c for c in self.feature_cols if c not in cat_cols]

        # Use only non-categorical targets for OLS
        numeric_targets = [c for c in self.target_cols if c not in cat_cols]

        if len(numeric_targets) == 0:
            # Nothing numeric to regress on – return empty/NA table
            return pd.DataFrame(
                columns=["cluster", "feature", "target", "beta", "stderr",
                         "t_value", "n_used", "coverage", "conf_label"]
            )

        records: List[Dict[str, Any]] = []
        N_total = len(df)

        for cid in self._clusters_:
            mask = self._select_core_mask(df, cid, has_soft)

            # Only keep numeric features + numeric targets in the regression subset
            subset = df.loc[mask, numeric_features + numeric_targets].copy()

            if self.drop_na:
                subset = subset.dropna(subset=numeric_features + numeric_targets)

            n_used = len(subset)

            # Too few rows – emit NA rows so formatter can show "insufficient"
            if n_used < max(8, len(numeric_features) + 3):
                for f in numeric_features:
                    for y in numeric_targets:
                        records.append(dict(
                            cluster=cid,
                            feature=f,
                            target=y,
                            beta=np.nan,
                            stderr=np.nan,
                            t_value=np.nan,
                            n_used=n_used,
                            coverage=round(n_used / max(N_total, 1), 4),
                            conf_label="insufficient data",
                        ))
                continue

            # Design matrix only on numeric features
            X, feature_names = self._prepare_design_matrix(
                subset[numeric_features].to_numpy(float)
            )
            beta_mat, se_mat, t_mat = self._ols_fit_multi(
                X,
                subset[numeric_targets].to_numpy(float),
            )

            # skip intercept row when adding records
            start = 1 if self.add_intercept else 0
            for j_feat, feat in enumerate(feature_names[start:], start=start):
                for j_t, tgt in enumerate(numeric_targets):
                    beta = float(beta_mat[j_feat, j_t])
                    se = float(se_mat[j_feat, j_t])
                    tval = float(t_mat[j_feat, j_t])

                    conf = self._conf_from_t(tval)

                    records.append(dict(
                        cluster=cid,
                        feature=feat,
                        target=tgt,
                        beta=round(beta, self.round_to),
                        stderr=round(se, self.round_to) if np.isfinite(se) else np.nan,
                        t_value=round(tval, self.round_to) if np.isfinite(tval) else np.nan,
                        n_used=int(n_used),
                        coverage=round(n_used / max(N_total, 1), 4),
                        conf_label=conf,
                    ))

        return pd.DataFrame.from_records(records)
    
    

    def compute_local_variability(
                                    self,
                                    df: pd.DataFrame,
                                    feature_cols: List[str],
                                    *,
                                    drop_na: bool = True,
                                    round_to: int = 3
                                ) -> Dict[str, Dict[str, Any]]:
            """
            Compute LocalRange and UniqueVals for given feature columns.
        
            Returns:
              {
                "feature1": {"LocalRange": float, "UniqueVals": int},
                "feature2": {...},
                ...
              }
            """
            if drop_na:
                df = df.dropna(subset=feature_cols)
        
            results: Dict[str, Dict[str, Any]] = {}
        
            for f in feature_cols:
                series = df[f]
                local_range = series.max() - series.min()
                unique_vals = series.nunique(dropna=drop_na)
        
                results[f] = {
                    "LocalRange": round(float(local_range), round_to),
                    "UniqueVals": int(unique_vals)
                }
        
            return results

    def _select_core_mask(self, df: pd.DataFrame, cid: int, has_soft: bool) -> np.ndarray:
        """Choose rows that define the core of a cluster (soft core or hard labels)."""
        if self.use_soft_core and has_soft:
            m = df[f"{self.soft_prefix}{cid}"] >= self.core_threshold
            if m.sum() < self.min_core:
                # fallback to hard labels
                m = (df[self.cluster_col] == cid)
        else:
            m = (df[self.cluster_col] == cid)
        return m.values

    def _prepare_design_matrix(self, X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Standardize (optional) and add intercept (optional)."""
        if self.standardize_X:
            mu = X.mean(axis=0, keepdims=True)
            sd = X.std(axis=0, ddof=1, keepdims=True)
            sd_safe = np.where(sd == 0, 1.0, sd)
            X = (X - mu) / sd_safe
        feature_names = list(self.feature_cols)
        if self.add_intercept:
            X = np.column_stack([np.ones((X.shape[0], 1)), X])
            feature_names = ["Intercept"] + feature_names
        return X, feature_names

    def _ols_fit_multi(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        OLS for multiple targets at once:
          beta = (X'X)^-1 X'Y ;  se_j = sqrt( diag((X'X)^-1) * sigma2_target )
        Returns (beta_mat, se_mat, t_mat)
        """
        XtX = X.T @ X
        XtX_inv = np.linalg.pinv(XtX)
        beta = XtX_inv @ (X.T @ Y)       # (p, m)
        Y_hat = X @ beta
        resid = Y - Y_hat

        dof = X.shape[0] - X.shape[1]
        if dof <= 0:
            sigma2 = np.var(resid, axis=0, ddof=1)
        else:
            sigma2 = (resid**2).sum(axis=0) / max(dof, 1)

        XtX_diag = np.diag(XtX_inv)
        XtX_diag = np.clip(XtX_diag, 0.0, None)
        se = np.sqrt(np.outer(XtX_diag, np.maximum(sigma2, 1e-12)))  # (p, m)

        with np.errstate(divide="ignore", invalid="ignore"):
            t = beta / np.where(se == 0, np.nan, se)

        return beta, se, t

    def _conf_from_t(self, tval: float) -> str:
        """Map |t| to high/med/low confidence."""
        if not np.isfinite(tval):
            return "low conf"
        at = abs(tval)
        if at >= 3.0: return "high conf"
        if at >= 2.0: return "med conf"
        return "low conf"

    # ===================== FORMATTING =====================

    def _format_feature_lines_for_cluster(
        self,
        effects_df: pd.DataFrame,
        variability: Dict[str, Dict[str, Any]],
        cluster_id: int,
        feature_order: Optional[List[str]] = None,
        target_order: Optional[List[str]] = None,
        round_to: Optional[int] = None) -> List[str]:
        
        """
        Build lines like:
          - UL_bitrate: +38.028 DL_bitrate_predicted / +0.004 RSRP_predicted / +0.002 SNR_predicted per +1 (high conf) [LocalRange: 2.000, UniqueVals: 2]
        where `variability[feat] = {"LocalRange": float, "UniqueVals": int}`.
        """
        sub = effects_df[effects_df["cluster"] == cluster_id].copy()
        if sub.empty:
            return [f"(Cluster {cluster_id}: insufficient data)"]

        # Rounding precision
        rto = self.round_to if round_to is None else round_to

        # Optional ordering
        if feature_order is None:
            feature_order = getattr(self, "feature_cols", sorted(sub["feature"].unique().tolist()))
        if target_order is None:
            target_order = getattr(self, "target_cols", sorted(sub["target"].unique().tolist()))

        sub["__f_ord"] = sub["feature"].map({f: i for i, f in enumerate(feature_order)}).fillna(1e9)
        sub["__t_ord"] = sub["target"].map({t: i for i, t in enumerate(target_order)}).fillna(1e9)
        sub = sub.sort_values(["__f_ord", "feature", "__t_ord", "target"])

        # Confidence ranking (conservative aggregation across targets)
        rank = {"high conf": 2, "med conf": 1, "low conf": 0}

        lines: List[str] = []
        for feat, g in sub.groupby("feature", sort=False):
            g_work = g
            if getattr(self, "remove_negative", False):
                g_work = g_work[g_work["beta"] >= 0]

            if g_work.empty:
                continue

            parts, confs = [], []
            for _, r in g_work.iterrows():
                beta = float(r["beta"])
                sign = "+" if beta >= 0 else ""
                parts.append(f"{sign}{beta:.{rto}f} {r['target']}")
                confs.append(str(r.get("conf_label", "low conf")))

            conf = min(confs, key=lambda c: rank.get(c, 0)) if confs else "low conf"

            # Pull variability; fall back to NA if missing
            v = variability.get(feat, {})
            rng = v.get("LocalRange", None)
            uniq = v.get("UniqueVals", None)

            if rng is None or uniq is None:
                tail = " [LocalRange: NA, UniqueVals: NA]"
            else:
                tail = f" [LocalRange: {float(rng):.{rto}f}, UniqueVals: {int(uniq)}]"

            lines.append(f"- {feat}: " + " / ".join(parts) + f" per +1 ({conf})" + tail)

        if not lines:
            lines = [f"(Cluster {cluster_id}: no effects to report with current settings)"]

        return lines
   

        
        
                

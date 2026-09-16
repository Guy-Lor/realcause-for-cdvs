"""
Core experiment runner for the revised CDV experiment.

Two public entry points:
    run_single_seed_sepsis(...)   — Sepsis / RealCause experiment
    run_single_seed_synthetic(...)— Synthetic DGP experiment

Both return the same structured result dict, which is saved to a checkpoint
file after each seed by the calling notebook.
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from .seed_utils import derive_seeds
from .cdv_discovery import (
    compute_feature_patterns,
    discover_cdvs,
    fit_propensity_model,
    check_cdv_retention,
    route_cases,
    compute_support_stats,
    generate_random_partition,
)
from .feature_builder import (
    build_global_sentinel,
    build_global_missingness,
    build_global_missingness_cdv,
    build_global_missingness_cdv_counterfactual,
    build_cdv_local_features,
    build_random_partition_features,
)
from .estimators import build_estimator_grid, LinearSLearnerCDV
from .metrics import all_metrics
from .oracle import fit_oracle_and_predict

METHODS = [
    "GLOBAL_SENTINEL",
    "GLOBAL_MISSINGNESS",
    "GLOBAL_MISSINGNESS_CDV",
    "CDV_SEPARATE",
    "MATCHED_RANDOM_PARTITIONS",
]


# ============================================================
# INTERNAL FIT/PREDICT HELPERS
# ============================================================

def _fit_predict(estimator, X_train, t_train, y_train, X_test, t_test):
    """Fit one estimator and return (ite_pred, error_msg)."""
    try:
        e = deepcopy(estimator)
        e.fit(X_train, t_train.flatten(), y_train.flatten())
        y0 = np.asarray(e.predict_outcome(X_test, np.zeros_like(t_test.flatten()))).flatten()
        y1 = np.asarray(e.predict_outcome(X_test, np.ones_like(t_test.flatten()))).flatten()
        return y1 - y0, None
    except Exception as ex:
        return np.full(len(X_test), np.nan), str(ex)


def _method_features_train(method, df, w_cols, cdv_assignment, retained_info, config, learner_name=""):
    """Build training feature matrix for a global method."""
    sentinel = config["SENTINEL_VALUE"]
    retained_cdv_ids = list(retained_info.keys())
    is_linear = learner_name == "S_Linear"

    if method == "GLOBAL_SENTINEL":
        return build_global_sentinel(df, w_cols)
    if method == "GLOBAL_MISSINGNESS":
        return build_global_missingness(df, w_cols, sentinel)
    if method == "GLOBAL_MISSINGNESS_CDV":
        return build_global_missingness_cdv(
            df, w_cols, cdv_assignment, retained_cdv_ids,
            sentinel, for_linear_s_learner=is_linear, t_col="t"
        )
    raise ValueError(f"Unknown global method: {method}")


def _method_features_test(method, df, w_cols, cdv_assignment, retained_info, config, learner_name=""):
    """Build test feature matrix for a global method (no T available)."""
    sentinel = config["SENTINEL_VALUE"]
    retained_cdv_ids = list(retained_info.keys())
    is_linear = learner_name == "S_Linear"

    if method == "GLOBAL_SENTINEL":
        return build_global_sentinel(df, w_cols)
    if method == "GLOBAL_MISSINGNESS":
        return build_global_missingness(df, w_cols, sentinel)
    if method == "GLOBAL_MISSINGNESS_CDV":
        if is_linear:
            # At predict time, interaction columns are handled inside LinearSLearnerCDV.
            # Return the base features WITHOUT interaction terms; the estimator adds them.
            return build_global_missingness_cdv(
                df, w_cols, cdv_assignment, retained_cdv_ids, sentinel,
                for_linear_s_learner=False  # no interactions in X_test; CDV object handles them
            )
        return build_global_missingness_cdv(
            df, w_cols, cdv_assignment, retained_cdv_ids, sentinel,
            for_linear_s_learner=False
        )
    raise ValueError(f"Unknown global method: {method}")


def _run_global_method(method, df_train, df_test, w_cols,
                       cdv_assignment_train, cdv_assignment_test,
                       retained_info, estimator_grid, config, ite_true_test):
    """Fit all learners for one global method. Returns {learner: pred_dict}."""
    results = {}
    retained_cdv_ids = list(retained_info.keys())
    n_cdv_cols = len(retained_cdv_ids) + 1  # +1 for OTHER

    for lname, base_est in estimator_grid.items():
        # Build training features
        if method == "GLOBAL_MISSINGNESS_CDV" and lname == "S_Linear":
            X_tr, _ = build_global_missingness_cdv(
                df_train, w_cols, cdv_assignment_train, retained_cdv_ids,
                config["SENTINEL_VALUE"], for_linear_s_learner=False
            )
            estimator = LinearSLearnerCDV(n_cdv_cols=n_cdv_cols)
        else:
            X_tr, _ = _method_features_train(
                method, df_train, w_cols, cdv_assignment_train, retained_info, config, lname
            )
            estimator = base_est

        t_tr = df_train["t"].values
        y_tr = df_train["y"].values

        # Build test features
        if method == "GLOBAL_MISSINGNESS_CDV" and lname == "S_Linear":
            X_te, _ = build_global_missingness_cdv(
                df_test, w_cols, cdv_assignment_test, retained_cdv_ids,
                config["SENTINEL_VALUE"], for_linear_s_learner=False
            )
        else:
            X_te, _ = _method_features_test(
                method, df_test, w_cols, cdv_assignment_test, retained_info, config, lname
            )

        t_te = df_test["t"].values
        ite_pred, err = _fit_predict(estimator, X_tr, t_tr, y_tr, X_te, t_te)

        if err:
            print(f"  [{method}|{lname}] FAILED: {err}")

        results[lname] = {
            "ite_pred": ite_pred,
            "ite_true": ite_true_test,
            "variant": cdv_assignment_test.values,
            "error": err,
        }
    return results


def _run_cdv_separate(df_train, df_test, w_cols,
                      cdv_assignment_train, cdv_assignment_test,
                      retained_info, estimator_grid, config, ite_true_test,
                      bootstrap_seeds):
    """CDV_SEPARATE: per-CDV model + global fallback for OTHER.

    When N_CDV_BOOTSTRAP > 1 each CDV model is an ensemble of bootstrap-resampled
    fits, giving the same variance reduction as MATCHED_RANDOM_PARTITIONS.
    """
    n_bootstrap = int(config.get("N_CDV_BOOTSTRAP", 1))
    results = {}
    n_test = len(df_test)

    for lname, base_est in estimator_grid.items():
        boot_preds = []

        for b_idx in range(n_bootstrap):
            ite_pred_b = np.full(n_test, np.nan)

            if n_bootstrap > 1:
                rng = np.random.default_rng(bootstrap_seeds[b_idx])
                boot_idx = rng.integers(0, len(df_train), size=len(df_train))
                df_tr = df_train.iloc[boot_idx].reset_index(drop=True)
                cdv_tr = cdv_assignment_train.iloc[boot_idx].reset_index(drop=True)
            else:
                df_tr = df_train
                cdv_tr = cdv_assignment_train

            # Global fallback — trained on ALL bootstrap training data.
            X_all, _ = build_global_sentinel(df_tr, w_cols)
            global_est = deepcopy(base_est)
            try:
                global_est.fit(X_all, df_tr["t"].values, df_tr["y"].values)
            except Exception as ex:
                print(f"  [CDV_SEPARATE|{lname}|b={b_idx}] Global fallback fit FAILED: {ex}")
                boot_preds.append(ite_pred_b)
                continue

            # OTHER test cases → global fallback
            other_mask = (cdv_assignment_test == "OTHER").values
            if other_mask.any():
                X_oth, _ = build_global_sentinel(df_test[other_mask], w_cols)
                t_oth = df_test["t"].values[other_mask]
                try:
                    y0 = np.asarray(global_est.predict_outcome(X_oth, np.zeros_like(t_oth))).flatten()
                    y1 = np.asarray(global_est.predict_outcome(X_oth, np.ones_like(t_oth))).flatten()
                    ite_pred_b[other_mask] = y1 - y0
                except Exception as ex:
                    print(f"  [CDV_SEPARATE|{lname}|b={b_idx}] OTHER predict FAILED: {ex}")

            # Per-CDV models
            for cdv_id, cdv_info in retained_info.items():
                pattern = cdv_info["pattern"]
                cdv_mask_tr = (cdv_tr == cdv_id).values
                cdv_mask_te = (cdv_assignment_test == cdv_id).values

                if not cdv_mask_tr.any():
                    if cdv_mask_te.any():
                        X_fb, _ = build_global_sentinel(df_test[cdv_mask_te], w_cols)
                        t_fb = df_test["t"].values[cdv_mask_te]
                        try:
                            y0 = np.asarray(global_est.predict_outcome(X_fb, np.zeros_like(t_fb))).flatten()
                            y1 = np.asarray(global_est.predict_outcome(X_fb, np.ones_like(t_fb))).flatten()
                            ite_pred_b[cdv_mask_te] = y1 - y0
                        except Exception:
                            pass
                    continue

                df_cdv_tr = df_tr[cdv_mask_tr]
                # After bootstrap, arm sizes may drop; fall back to global if too small.
                n_t = int((df_cdv_tr["t"] == 1).sum())
                n_c = int((df_cdv_tr["t"] == 0).sum())
                if n_t < 2 or n_c < 2:
                    if cdv_mask_te.any():
                        X_fb, _ = build_global_sentinel(df_test[cdv_mask_te], w_cols)
                        t_fb = df_test["t"].values[cdv_mask_te]
                        try:
                            y0 = np.asarray(global_est.predict_outcome(X_fb, np.zeros_like(t_fb))).flatten()
                            y1 = np.asarray(global_est.predict_outcome(X_fb, np.ones_like(t_fb))).flatten()
                            ite_pred_b[cdv_mask_te] = y1 - y0
                        except Exception:
                            pass
                    continue

                X_cdv_tr, _ = build_cdv_local_features(df_cdv_tr, w_cols, pattern)
                cdv_est = deepcopy(base_est)
                try:
                    cdv_est.fit(X_cdv_tr, df_cdv_tr["t"].values, df_cdv_tr["y"].values)
                except Exception as ex:
                    print(f"  [CDV_SEPARATE|{lname}|b={b_idx}|{cdv_id}] Fit FAILED: {ex}")
                    if cdv_mask_te.any():
                        X_fb, _ = build_global_sentinel(df_test[cdv_mask_te], w_cols)
                        t_fb = df_test["t"].values[cdv_mask_te]
                        try:
                            y0 = np.asarray(global_est.predict_outcome(X_fb, np.zeros_like(t_fb))).flatten()
                            y1 = np.asarray(global_est.predict_outcome(X_fb, np.ones_like(t_fb))).flatten()
                            ite_pred_b[cdv_mask_te] = y1 - y0
                        except Exception:
                            pass
                    continue

                if cdv_mask_te.any():
                    df_cdv_te = df_test[cdv_mask_te]
                    X_cdv_te, _ = build_cdv_local_features(df_cdv_te, w_cols, pattern)
                    t_te = df_cdv_te["t"].values
                    try:
                        y0 = np.asarray(cdv_est.predict_outcome(X_cdv_te, np.zeros_like(t_te))).flatten()
                        y1 = np.asarray(cdv_est.predict_outcome(X_cdv_te, np.ones_like(t_te))).flatten()
                        ite_pred_b[cdv_mask_te] = y1 - y0
                    except Exception as ex:
                        print(f"  [CDV_SEPARATE|{lname}|b={b_idx}|{cdv_id}] Predict FAILED: {ex}")

            boot_preds.append(ite_pred_b)

        mean_pred = np.nanmean(np.stack(boot_preds, axis=0), axis=0) if boot_preds else np.full(n_test, np.nan)
        results[lname] = {
            "ite_pred": mean_pred,
            "ite_true": ite_true_test,
            "variant": cdv_assignment_test.values,
            "error": None,
        }
    return results


def _run_matched_random_partitions(
    df_train, df_test, w_cols,
    cdv_assignment_train, cdv_assignment_test,
    retained_info, estimator_grid, config, ite_true_test, perm_seeds
):
    """
    MATCHED_RANDOM_PARTITIONS: N permutations, aggregate by averaging ite_pred.
    Test routing follows CDV slot assignment (same-slot principle).
    Returns {learner: pred_dict with aggregated ite_pred + raw perm list}.
    """
    n_perms = config["N_RANDOM_PERMUTATIONS"]
    n_test = len(df_test)
    sentinel = config["SENTINEL_VALUE"]

    # Per-learner storage: list of ite_pred arrays, one per permutation
    perm_ite_preds = {lname: [] for lname in estimator_grid}

    for perm_idx in range(n_perms):
        perm_seed = perm_seeds[perm_idx]
        perm_assignment = generate_random_partition(
            df_train, retained_info, cdv_assignment_train, perm_seed
        )

        for lname, base_est in estimator_grid.items():
            ite_pred = np.full(n_test, np.nan)

            # Global fallback for OTHER slot (same as CDV_SEPARATE)
            X_all, _ = build_random_partition_features(df_train, w_cols, sentinel)
            global_est = deepcopy(base_est)
            try:
                global_est.fit(X_all, df_train["t"].values, df_train["y"].values)
            except Exception:
                perm_ite_preds[lname].append(ite_pred)
                continue

            # OTHER test cases → global model
            other_mask_te = (cdv_assignment_test == "OTHER").values
            if other_mask_te.any():
                X_oth, _ = build_random_partition_features(df_test[other_mask_te], w_cols, sentinel)
                t_oth = df_test["t"].values[other_mask_te]
                try:
                    y0 = np.asarray(global_est.predict_outcome(X_oth, np.zeros_like(t_oth))).flatten()
                    y1 = np.asarray(global_est.predict_outcome(X_oth, np.ones_like(t_oth))).flatten()
                    ite_pred[other_mask_te] = y1 - y0
                except Exception:
                    pass

            # Per-random-group models
            for cdv_id in retained_info:
                mask_tr = (perm_assignment == cdv_id).values
                mask_te = (cdv_assignment_test == cdv_id).values  # same-slot routing

                if not mask_tr.any():
                    if mask_te.any():
                        X_fb, _ = build_random_partition_features(df_test[mask_te], w_cols, sentinel)
                        t_fb = df_test["t"].values[mask_te]
                        try:
                            y0 = np.asarray(global_est.predict_outcome(X_fb, np.zeros_like(t_fb))).flatten()
                            y1 = np.asarray(global_est.predict_outcome(X_fb, np.ones_like(t_fb))).flatten()
                            ite_pred[mask_te] = y1 - y0
                        except Exception:
                            pass
                    continue

                df_grp_tr = df_train[mask_tr]
                X_grp, _ = build_random_partition_features(df_grp_tr, w_cols, sentinel)
                grp_est = deepcopy(base_est)
                try:
                    grp_est.fit(X_grp, df_grp_tr["t"].values, df_grp_tr["y"].values)
                except Exception:
                    if mask_te.any():
                        X_fb, _ = build_random_partition_features(df_test[mask_te], w_cols, sentinel)
                        t_fb = df_test["t"].values[mask_te]
                        try:
                            y0 = np.asarray(global_est.predict_outcome(X_fb, np.zeros_like(t_fb))).flatten()
                            y1 = np.asarray(global_est.predict_outcome(X_fb, np.ones_like(t_fb))).flatten()
                            ite_pred[mask_te] = y1 - y0
                        except Exception:
                            pass
                    continue

                if mask_te.any():
                    X_grp_te, _ = build_random_partition_features(df_test[mask_te], w_cols, sentinel)
                    t_te = df_test["t"].values[mask_te]
                    try:
                        y0 = np.asarray(grp_est.predict_outcome(X_grp_te, np.zeros_like(t_te))).flatten()
                        y1 = np.asarray(grp_est.predict_outcome(X_grp_te, np.ones_like(t_te))).flatten()
                        ite_pred[mask_te] = y1 - y0
                    except Exception:
                        pass

            perm_ite_preds[lname].append(ite_pred)

    # Aggregate across permutations: mean ite_pred per test case
    results = {}
    for lname in estimator_grid:
        preds = perm_ite_preds[lname]
        if preds:
            mean_pred = np.nanmean(np.stack(preds, axis=0), axis=0)
        else:
            mean_pred = np.full(n_test, np.nan)
        results[lname] = {
            "ite_pred": mean_pred,
            "ite_true": ite_true_test,
            "variant": cdv_assignment_test.values,
            "perm_ite_preds": preds,  # raw permutation predictions for detailed analysis
            "error": None,
        }
    return results


# ============================================================
# ORACLE RUNNER
# ============================================================

def _run_oracle_for_all_methods(
    df_train, df_test, w_cols,
    cdv_assignment_train, cdv_assignment_test,
    retained_info, estimator_grid, config,
    true_cate_train, true_cate_test,
    oracle_seed
):
    """Run oracle learner selection for every method. Returns {method: oracle_dict}."""
    n_folds = config["N_ORACLE_CV_FOLDS"]
    sentinel = config["SENTINEL_VALUE"]
    retained_cdv_ids = list(retained_info.keys())
    oracle_results = {}

    for method in METHODS:
        if method == "CDV_SEPARATE":
            # Oracle for CDV_SEPARATE: select best learner using pooled training data
            X_tr, _ = build_global_missingness(df_train, w_cols, sentinel)
            X_te, _ = build_global_missingness(df_test, w_cols, sentinel)
        elif method == "MATCHED_RANDOM_PARTITIONS":
            X_tr, _ = build_global_sentinel(df_train, w_cols)
            X_te, _ = build_global_sentinel(df_test, w_cols)
        elif method == "GLOBAL_MISSINGNESS_CDV":
            # Build without S_Linear interactions for oracle (RF-based selection only)
            X_tr, _ = build_global_missingness_cdv(
                df_train, w_cols, cdv_assignment_train, retained_cdv_ids,
                sentinel, for_linear_s_learner=False
            )
            X_te, _ = build_global_missingness_cdv(
                df_test, w_cols, cdv_assignment_test, retained_cdv_ids,
                sentinel, for_linear_s_learner=False
            )
        else:
            X_tr, _ = _method_features_train(
                method, df_train, w_cols, cdv_assignment_train, retained_info, config
            )
            X_te, _ = _method_features_test(
                method, df_test, w_cols, cdv_assignment_test, retained_info, config
            )

        # Exclude S_Linear for oracle (different dimensionality in CDV method)
        oracle_grid = {k: v for k, v in estimator_grid.items() if k != "S_Linear"}

        oracle = fit_oracle_and_predict(
            X_tr, df_train["t"].values, df_train["y"].values,
            true_cate_train, X_te, df_test["t"].values,
            oracle_grid, n_folds, oracle_seed
        )
        oracle["ite_true_test"] = true_cate_test
        oracle["metrics"] = all_metrics(oracle["ite_pred_test"], true_cate_test)
        oracle_results[method] = oracle

    return oracle_results


# ============================================================
# CORE SEED RUNNER
# ============================================================

class TarNetPropensityWrapper:
    """
    Sklearn predict_proba-compatible wrapper around TarNet's propensity head.
    Uses the same clipped propensities that generated T, making the retention
    check fully consistent with the DGP.
    """
    def __init__(self, tarnet_model, positivity_clip: float = 0.0):
        self._model = tarnet_model
        self._clip = positivity_clip

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        X_t = torch.tensor(X.astype(np.float32))
        X_t = self._model.w_transform.transform(X_t)
        with torch.no_grad():
            logit = self._model.mlp_t_w(X_t)
        p = torch.sigmoid(logit).cpu().numpy().flatten()
        if self._clip > 0:
            p = np.clip(p, self._clip, 1.0 - self._clip)
        return np.column_stack([1.0 - p, p])


# ============================================================

def _run_seed_core(
    outer_seed: int,
    config: dict,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    w_cols: list,
    true_cate_fn,      # callable(w_array) -> cate_array, or None (use df['ite'])
    seeds: dict,
    prop_model_override=None,  # if set, skips logistic-regression fitting
):
    """
    Core per-seed logic shared by Sepsis and Synthetic runners.

    Assumes df_train and df_test already contain columns: w_cols, t, y
    and optionally 'ite' (a fallback target when true_cate_fn is not supplied).
    """
    print(f"\n[Seed {outer_seed}] Starting core runner...")

    # ── Propensity model ────────────────────────────────────────────────────
    X_full_train = df_train[w_cols].values.astype(np.float64)
    t_train = df_train["t"].values.flatten()
    if prop_model_override is not None:
        prop_model = prop_model_override
    else:
        prop_model = fit_propensity_model(X_full_train, t_train, random_state=seeds["model"])

    # ── CDV discovery (train only) ──────────────────────────────────────────
    discovered_patterns, coverage_df = discover_cdvs(
        df_train, w_cols, config["SENTINEL_VALUE"], config["CDV_COVERAGE_THRESHOLD"]
    )

    print(f"[Seed {outer_seed}] Discovered {len(discovered_patterns)} CDV patterns.")

    # ── Retention checks ────────────────────────────────────────────────────
    train_patterns = compute_feature_patterns(df_train, w_cols, config["SENTINEL_VALUE"])
    retained_info = {}
    violations_log = {}

    for i, pattern in enumerate(discovered_patterns):
        cdv_id = f"CDV_{i + 1}"
        mask = train_patterns == pattern
        cdv_df = df_train[mask].copy()
        retained, viols = check_cdv_retention(cdv_df, w_cols, prop_model, config)
        if retained:
            retained_info[cdv_id] = {
                "pattern": pattern,
                "n": len(cdv_df),
                "n_treated": int((cdv_df["t"] == 1).sum()),
                "n_control": int((cdv_df["t"] == 0).sum()),
            }
        else:
            violations_log[cdv_id] = {"pattern": pattern, "violations": viols}
            print(f"[Seed {outer_seed}] {cdv_id} FAILED retention: {viols}")

    print(f"[Seed {outer_seed}] Retained {len(retained_info)} CDVs; "
          f"{len(violations_log)} sent to OTHER due to retention failure.")

    # ── Sanity checks ────────────────────────────────────────────────────────
    train_ids = set(df_train.index)
    test_ids = set(df_test.index)
    assert train_ids.isdisjoint(test_ids), "SANITY: train and test case IDs overlap!"

    # ── Case routing ────────────────────────────────────────────────────────
    cdv_train = route_cases(df_train, retained_info, w_cols, config["SENTINEL_VALUE"])
    cdv_test = route_cases(df_test, retained_info, w_cols, config["SENTINEL_VALUE"])

    # Every test case must be assigned to exactly one slot
    assert cdv_test.notna().all(), "SANITY: some test cases have NaN CDV assignment!"

    # ── True CATE targets ────────────────────────────────────────────────────
    if true_cate_fn is not None:
        print(f"[Seed {outer_seed}] Computing oracle CATE via true_cate_fn...")
        ite_true_test = true_cate_fn(df_test[w_cols].values.astype(np.float64))
        ite_true_train = true_cate_fn(df_train[w_cols].values.astype(np.float64))
    else:
        ite_true_test = df_test["ite"].values.flatten()
        ite_true_train = df_train["ite"].values.flatten()

    # ── Build estimator grid ─────────────────────────────────────────────────
    estimator_grid = build_estimator_grid(config, seeds["model"])

    # ── Run all 5 methods ───────────────────────────────────────────────────
    predictions = {}

    for method in METHODS:
        print(f"[Seed {outer_seed}] Running method: {method}...")
        if method == "CDV_SEPARATE":
            predictions[method] = _run_cdv_separate(
                df_train, df_test, w_cols, cdv_train, cdv_test,
                retained_info, estimator_grid, config, ite_true_test,
                seeds["cdv_bootstrap"]
            )
        elif method == "MATCHED_RANDOM_PARTITIONS":
            predictions[method] = _run_matched_random_partitions(
                df_train, df_test, w_cols, cdv_train, cdv_test,
                retained_info, estimator_grid, config, ite_true_test,
                seeds["permutations"]
            )
        else:
            predictions[method] = _run_global_method(
                method, df_train, df_test, w_cols, cdv_train, cdv_test,
                retained_info, estimator_grid, config, ite_true_test
            )

    # ── Compute metrics ──────────────────────────────────────────────────────
    metrics_summary = {}
    for method, ldict in predictions.items():
        metrics_summary[method] = {}
        for lname, pred in ldict.items():
            ip = pred.get("ite_pred", np.array([]))
            it = pred.get("ite_true", np.array([]))
            metrics_summary[method][lname] = all_metrics(ip, it)

    # ── Oracle selection ─────────────────────────────────────────────────────
    print(f"[Seed {outer_seed}] Running oracle selection...")
    oracle_results = _run_oracle_for_all_methods(
        df_train, df_test, w_cols, cdv_train, cdv_test,
        retained_info, estimator_grid, config,
        ite_true_train, ite_true_test, seeds["oracle_cv"]
    )

    # ── Support statistics ───────────────────────────────────────────────────
    support_stats = compute_support_stats(df_train, cdv_train, prop_model, w_cols, config)

    pct_other_train = float((cdv_train == "OTHER").mean() * 100)
    pct_other_test = float((cdv_test == "OTHER").mean() * 100)

    return {
        "outer_seed": outer_seed,
        "seeds_used": seeds,
        "n_train": len(df_train),
        "n_test": len(df_test),
        "retained_cdv_info": retained_info,
        "coverage_df": coverage_df,
        "violations_log": violations_log,
        "pct_other_train": pct_other_train,
        "pct_other_test": pct_other_test,
        "cdv_assignment_test": cdv_test.values,
        "ite_true_test": ite_true_test,
        "predictions": predictions,
        "metrics": metrics_summary,
        "oracle": oracle_results,
        "support_stats": support_stats,
    }


# ============================================================
# SEPSIS ENTRY POINT
# ============================================================

def _sample_sepsis_data(
    realcause_model, w_raw: np.ndarray, draw_seed: int,
    positivity_clip: float = 0.0,
) -> tuple:
    """
    Draw stochastic T, Y0, Y1 for all Sepsis cases from the fitted RealCause model.
    W is fixed (from the CSV). Only T and Y change per seed.

    positivity_clip: if > 0, propensity scores are clipped to
        [positivity_clip, 1 - positivity_clip] before sampling T.
        E.g. positivity_clip=0.1 enforces p(T=1|W) ∈ [0.10, 0.90].
    Returns (t_all, y_obs_all, y0_all, y1_all).
    """
    import torch
    realcause_model.set_seed(draw_seed)

    if positivity_clip > 0:
        w_tensor = torch.tensor(w_raw.astype(np.float32))
        w_transformed = realcause_model.w_transform.transform(w_tensor)
        with torch.no_grad():
            logit = realcause_model.mlp_t_w(w_transformed)
        p = torch.sigmoid(logit).cpu().numpy().flatten()
        p_clipped = np.clip(p, positivity_clip, 1.0 - positivity_clip)
        t_flat = (p_clipped > np.random.rand(len(p_clipped))).astype(np.float32)

        t_internal = t_flat.reshape(-1, 1)
        w_np = w_transformed.numpy()
        # _sample_y already clips to [outcome_min, outcome_max] before returning
        y0_flat, y1_flat = realcause_model._sample_y(t_internal, w_np, ret_counterfactuals=True)
        # untransform maps from normalized [0,1] → original y scale (e.g. cycle-time minutes)
        y0_flat = realcause_model.y_transform.untransform(y0_flat).flatten()
        y1_flat = realcause_model.y_transform.untransform(y1_flat).flatten()
    else:
        w_out, t_out, (y0_out, y1_out) = realcause_model.sample(
            w=w_raw.astype(np.float32),
            transform_w=True,
            untransform=True,
            seed=None,  # seed already set via set_seed above
            overlap=1,
            ret_counterfactuals=True,
        )
        t_flat = np.asarray(t_out).flatten()
        y0_flat = np.asarray(y0_out).flatten()
        y1_flat = np.asarray(y1_out).flatten()

    y_obs = np.where(t_flat == 1, y1_flat, y0_flat)
    return t_flat, y_obs, y0_flat, y1_flat


def run_single_seed_sepsis(
    outer_seed: int,
    config: dict,
    realcause_model,
    df_cases: pd.DataFrame,
    w_cols: list,
    true_cate_fn,
) -> dict:
    """
    Run one outer seed of the revised Sepsis experiment.

    Parameters
    ----------
    outer_seed : int
    config : dict  (from sepsis/config.py)
    realcause_model : fitted TarNet
    df_cases : pd.DataFrame  (all 810 sepsis cases, NaN already present in feature cols)
    w_cols : list of feature column names
    true_cate_fn : callable(w_array) -> cate_array
        e.g. lambda w: estimate_sigmoid_flow_cate(model, w)
    """
    seeds = derive_seeds(outer_seed)

    # 1. Draw stochastic T, Y for all cases using fixed W
    w_raw = df_cases[w_cols].fillna(-1).values.astype(np.float64)
    positivity_clip = config.get("POSITIVITY_CLIP", 0.0)
    t_all, y_obs_all, y0_all, y1_all = _sample_sepsis_data(
        realcause_model, w_raw, seeds["draw_train"], positivity_clip=positivity_clip
    )

    # 2. Build full DataFrame (sentinel-filled W + stochastic T, Y)
    df_all = df_cases[w_cols].fillna(-1).copy().reset_index(drop=True)
    df_all["t"] = t_all
    df_all["y"] = y_obs_all
    df_all["y0"] = y0_all
    df_all["y1"] = y1_all

    # 3. Stratified 70/30 split on treatment
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(
        np.arange(len(df_all)),
        test_size=config["TEST_PROP"],
        stratify=t_all.astype(int),
        random_state=seeds["split"],
    )

    df_train = df_all.iloc[train_idx].copy()
    df_test = df_all.iloc[test_idx].copy()

    # Sanity: no overlap
    assert len(set(train_idx) & set(test_idx)) == 0, "Train/test overlap detected!"

    print(f"[Seed {outer_seed}] Split: {len(df_train)} train / {len(df_test)} test "
          f"(treatment rates: {df_train['t'].mean():.3f} / {df_test['t'].mean():.3f})")

    # Use TarNet's own propensity head for retention checks (consistent with DGP)
    tarnet_prop = TarNetPropensityWrapper(
        realcause_model, positivity_clip=config.get("POSITIVITY_CLIP", 0.0)
    )
    return _run_seed_core(
        outer_seed, config, df_train, df_test, w_cols, true_cate_fn, seeds,
        prop_model_override=tarnet_prop,
    )


# ============================================================
# SYNTHETIC ENTRY POINT
# ============================================================

def run_single_seed_synthetic(
    outer_seed: int,
    config: dict,
    alpha: float,
    w_cols: list,
) -> dict:
    """
    Run one outer seed of the revised Synthetic experiment.

    Generates FRESH train and test datasets from the DGP per seed.
    true_cate_fn computes the pooled CATE conditional on observed features.
    """
    import sys, os
    sys.path.insert(0, _PROJECT_ROOT)
    from cdv_utils.synthetic_dgp import generate_synthetic_dataset, observed_history_cate

    seeds = derive_seeds(outer_seed)

    n_train = config["N_TRAIN"]
    n_test = config["N_TEST"]

    print(f"[Seed {outer_seed}|α={alpha}] Generating train/test data...")
    df_train_raw = generate_synthetic_dataset(n=n_train, alpha=alpha, seed=seeds["draw_train"])
    df_test_raw = generate_synthetic_dataset(n=n_test, alpha=alpha, seed=seeds["draw_test"])

    # Keep only needed columns
    needed = w_cols + ["t", "y", "y0", "y1", "ite"]
    df_train = df_train_raw[needed].copy().reset_index(drop=True)
    df_test = df_test_raw[needed].copy().reset_index(drop=True)
    # Train/test are independently-drawn datasets, not a split of one dataset,
    # so their reset indices would otherwise both start at 0 and collide.
    df_test.index = df_test.index + len(df_train)

    # The generator's ite remains the individual effect; it is not the score target.
    def true_cate_fn(w_array):
        return observed_history_cate(w_array, alpha=alpha, w_cols=w_cols)

    return _run_seed_core(outer_seed, config, df_train, df_test, w_cols, true_cate_fn, seeds)


# ============================================================
# CHECKPOINT UTILITIES
# ============================================================

def load_checkpoint(path: str) -> dict:
    """Load saved results dict from pickle, or return {} if not found."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def save_checkpoint(results: dict, path: str):
    """Save results dict to pickle."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(results, f)


def run_outer_seed_loop_sepsis(
    outer_seeds, config, realcause_model, df_cases, w_cols, true_cate_fn,
    checkpoint_path
):
    """
    Outer seed loop for Sepsis. Saves a checkpoint after each seed.
    Skips already-completed seeds automatically.
    """
    results = load_checkpoint(checkpoint_path)
    completed = set(results.keys())
    print(f"Loaded {len(completed)} completed seeds from checkpoint.")

    for outer_seed in outer_seeds:
        if outer_seed in completed:
            print(f"[Seed {outer_seed}] Already completed, skipping.")
            continue
        try:
            seed_result = run_single_seed_sepsis(
                outer_seed, config, realcause_model, df_cases, w_cols, true_cate_fn
            )
            results[outer_seed] = seed_result
            save_checkpoint(results, checkpoint_path)
            print(f"[Seed {outer_seed}] ✓ Saved.")
        except Exception as ex:
            import traceback
            print(f"[Seed {outer_seed}] ✗ FAILED: {ex}")
            traceback.print_exc()

    return results


def run_outer_seed_loop_synthetic(
    outer_seeds, alpha_values, config, w_cols, checkpoint_path_template
):
    """
    Outer seed × alpha loop for Synthetic. Saves a checkpoint per alpha.
    checkpoint_path_template should contain '{alpha:.2f}', e.g.
    'artifacts/results_alpha_{alpha:.2f}.pkl'.
    """
    results_by_alpha = {}

    for alpha in alpha_values:
        path = checkpoint_path_template.format(alpha=alpha)
        results_by_alpha[alpha] = load_checkpoint(path)
        completed = set(results_by_alpha[alpha].keys())
        print(f"\n[α={alpha}] Loaded {len(completed)} completed seeds.")

        for outer_seed in outer_seeds:
            if outer_seed in completed:
                continue
            try:
                seed_result = run_single_seed_synthetic(outer_seed, config, alpha, w_cols)
                results_by_alpha[alpha][outer_seed] = seed_result
                save_checkpoint(results_by_alpha[alpha], path)
                print(f"[Seed {outer_seed}|α={alpha}] ✓ Saved.")
            except Exception as ex:
                import traceback
                print(f"[Seed {outer_seed}|α={alpha}] ✗ FAILED: {ex}")
                traceback.print_exc()

    return results_by_alpha

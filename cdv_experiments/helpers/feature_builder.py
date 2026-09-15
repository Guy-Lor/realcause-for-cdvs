"""
Feature matrix builders for the five compared methods.

Each builder takes a DataFrame and returns (X_array, feature_col_names).
All builders honour the sentinel value convention:
    feature >= 0  → present
    feature <  0  → structurally absent (sentinel)
"""
import numpy as np
import pandas as pd


# ============================================================
# METHOD 1: GLOBAL_SENTINEL
# ============================================================

def build_global_sentinel(df: pd.DataFrame, w_cols: list) -> tuple:
    """
    Union of all features with sentinel encoding as-is.
    No extra columns added.
    """
    X = df[w_cols].values.astype(np.float64)
    return X, list(w_cols)


# ============================================================
# METHOD 2: GLOBAL_MISSINGNESS
# ============================================================

def build_global_missingness(
    df: pd.DataFrame, w_cols: list, sentinel_value: float = -1
) -> tuple:
    """
    Union of features + binary indicator for every structurally absent feature.

    missing_{col} = 1 if feature < 0 (absent), 0 otherwise.
    """
    w = df[w_cols].values.astype(np.float64)
    missing_flags = (w < 0).astype(np.float64)

    X = np.concatenate([w, missing_flags], axis=1)
    missing_cols = [f"missing_{c}" for c in w_cols]
    feature_names = list(w_cols) + missing_cols
    return X, feature_names


# ============================================================
# METHOD 3: GLOBAL_MISSINGNESS_CDV
# ============================================================

def build_global_missingness_cdv(
    df: pd.DataFrame,
    w_cols: list,
    cdv_assignment: pd.Series,
    retained_cdv_ids: list,
    sentinel_value: float = -1,
    for_linear_s_learner: bool = False,
    t_col: str = "t",
) -> tuple:
    """
    Union of features + missingness indicators + CDV one-hot columns.

    When for_linear_s_learner=True, also adds T × is_CDV_k interaction terms
    so a linear model can represent CDV-specific treatment effects.

    The CDV assignment is passed as a Series aligned to df's index.
    """
    w = df[w_cols].values.astype(np.float64)
    missing_flags = (w < 0).astype(np.float64)
    missing_cols = [f"missing_{c}" for c in w_cols]

    # CDV one-hot columns
    cdv_one_hot_list = []
    cdv_one_hot_names = []
    for cdv_id in retained_cdv_ids:
        col = (cdv_assignment == cdv_id).astype(np.float64).values.reshape(-1, 1)
        cdv_one_hot_list.append(col)
        cdv_one_hot_names.append(f"is_{cdv_id}")
    other_col = (cdv_assignment == "OTHER").astype(np.float64).values.reshape(-1, 1)
    cdv_one_hot_list.append(other_col)
    cdv_one_hot_names.append("is_OTHER")

    cdv_block = np.concatenate(cdv_one_hot_list, axis=1)

    if not for_linear_s_learner:
        X = np.concatenate([w, missing_flags, cdv_block], axis=1)
        feature_names = list(w_cols) + missing_cols + cdv_one_hot_names
        return X, feature_names

    # For linear S-learner: add T × CDV interaction columns
    t_vals = df[t_col].values.flatten().astype(np.float64).reshape(-1, 1)
    interaction_block = cdv_block * t_vals
    interaction_names = [f"t_x_{n}" for n in cdv_one_hot_names]

    X = np.concatenate([w, missing_flags, cdv_block, interaction_block], axis=1)
    feature_names = list(w_cols) + missing_cols + cdv_one_hot_names + interaction_names
    return X, feature_names


def build_global_missingness_cdv_counterfactual(
    df: pd.DataFrame,
    w_cols: list,
    cdv_assignment: pd.Series,
    retained_cdv_ids: list,
    t_value: float,
    sentinel_value: float = -1,
) -> tuple:
    """
    Build GLOBAL_MISSINGNESS_CDV features for counterfactual prediction at fixed t.
    Used for the linear S-learner at predict time so interaction terms update correctly.
    """
    w = df[w_cols].values.astype(np.float64)
    missing_flags = (w < 0).astype(np.float64)
    missing_cols = [f"missing_{c}" for c in w_cols]

    cdv_one_hot_list = []
    cdv_one_hot_names = []
    for cdv_id in retained_cdv_ids:
        col = (cdv_assignment == cdv_id).astype(np.float64).values.reshape(-1, 1)
        cdv_one_hot_list.append(col)
        cdv_one_hot_names.append(f"is_{cdv_id}")
    other_col = (cdv_assignment == "OTHER").astype(np.float64).values.reshape(-1, 1)
    cdv_one_hot_list.append(other_col)
    cdv_one_hot_names.append("is_OTHER")
    cdv_block = np.concatenate(cdv_one_hot_list, axis=1)

    t_arr = np.full((len(df), 1), float(t_value))
    interaction_block = cdv_block * t_arr
    interaction_names = [f"t_x_{n}" for n in cdv_one_hot_names]

    X = np.concatenate([w, missing_flags, cdv_block, interaction_block], axis=1)
    feature_names = list(w_cols) + missing_cols + cdv_one_hot_names + interaction_names
    return X, feature_names


# ============================================================
# METHOD 4 (sub-helper): CDV_SEPARATE local feature set
# ============================================================

def build_cdv_local_features(
    df: pd.DataFrame, w_cols: list, cdv_pattern: str
) -> tuple:
    """
    Return only the columns that are present (bit='1') for this CDV's pattern.
    Used for the CDV_SEPARATE method.
    """
    present_cols = [w_cols[i] for i, bit in enumerate(cdv_pattern) if bit == "1"]
    if not present_cols:
        # Fallback: use all columns (degenerate case)
        present_cols = list(w_cols)
    X = df[present_cols].values.astype(np.float64)
    return X, present_cols


# ============================================================
# METHOD 5: MATCHED_RANDOM_PARTITIONS (same as GLOBAL_MISSINGNESS)
# ============================================================

def build_random_partition_features(
    df: pd.DataFrame, w_cols: list, sentinel_value: float = -1
) -> tuple:
    """
    Each random-partition model uses GLOBAL_SENTINEL representation, matching
    CDV_SEPARATE's fallback/per-group encoding so the placebo is a fair comparison.
    """
    return build_global_sentinel(df, w_cols)

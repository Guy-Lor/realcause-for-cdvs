"""
CDV Discovery, Retention, and Support Diagnostics.

All discovery and retention decisions use TRAINING DATA ONLY.
Test data is only used for routing (mapping patterns to retained CDV slots).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import warnings


# ============================================================
# PATTERN UTILITIES
# ============================================================

def compute_feature_patterns(df: pd.DataFrame, w_cols: list, sentinel_value: float = 0) -> pd.Series:
    """
    Compute a binary presence/absence pattern string for each row.
    bit_i = '1' if feature_i >= 0 (present), '0' otherwise (absent/sentinel).
    """
    mask = df[w_cols] >= 0
    return mask.apply(lambda r: "".join("1" if v else "0" for v in r), axis=1)


# ============================================================
# DISCOVERY
# ============================================================

def discover_cdvs(
    df_train: pd.DataFrame,
    w_cols: list,
    sentinel_value: float,
    coverage_threshold: float,
) -> tuple:
    """
    Greedy CDV discovery on training data.

    Sorts patterns by descending frequency, then adds patterns one by one
    until cumulative coverage first reaches or exceeds coverage_threshold.

    Returns
    -------
    retained_patterns : list of str
        The k* patterns that together cover >= coverage_threshold of training data.
    coverage_df : pd.DataFrame
        Full stats for all discovered patterns with columns:
        rank, pattern, count, pct, cumulative_pct, is_retained
    """
    patterns = compute_feature_patterns(df_train, w_cols, sentinel_value)
    pattern_counts = patterns.value_counts()
    total = len(df_train)

    cumulative = 0.0
    retained_patterns = []
    rows = []

    for rank, (pattern, count) in enumerate(pattern_counts.items(), start=1):
        pct = count / total
        cumulative += pct
        retained_patterns.append(pattern)
        rows.append({
            "rank": rank,
            "pattern": pattern,
            "count": int(count),
            "pct": round(pct * 100, 2),
            "cumulative_pct": round(cumulative * 100, 2),
            "is_retained": True,
        })
        if cumulative >= coverage_threshold:
            break

    # Remaining patterns all go to OTHER
    for rank2, (pattern, count) in enumerate(
        pattern_counts.iloc[len(retained_patterns):].items(),
        start=len(retained_patterns) + 1,
    ):
        rows.append({
            "rank": rank2,
            "pattern": pattern,
            "count": int(count),
            "pct": round(count / total * 100, 2),
            "cumulative_pct": None,
            "is_retained": False,
        })

    coverage_df = pd.DataFrame(rows)
    return retained_patterns, coverage_df


# ============================================================
# PROPENSITY MODEL
# ============================================================

def fit_propensity_model(X_train: np.ndarray, t_train: np.ndarray, random_state: int):
    """
    Fit a logistic regression propensity model on training data.
    Returns the fitted sklearn LogisticRegression.
    """
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=random_state,
        C=1.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, t_train.flatten())
    return model


# ============================================================
# RETENTION CHECKS
# ============================================================

def _check_option_a(t_arr: np.ndarray, overlap_lo: float, overlap_hi: float):
    """Option A: mean treatment rate in [overlap_lo, overlap_hi]."""
    mean_t = t_arr.mean()
    if mean_t < overlap_lo:
        return False, f"OptionA: mean_treatment_rate={mean_t:.3f} < lo={overlap_lo}"
    if mean_t > overlap_hi:
        return False, f"OptionA: mean_treatment_rate={mean_t:.3f} > hi={overlap_hi}"
    return True, None


def _check_option_b(
    X: np.ndarray,
    prop_model,
    overlap_lo: float,
    overlap_hi: float,
    overlap_min_fraction: float,
):
    """Option B: fraction of cases with p̂ in [lo, hi] >= overlap_min_fraction."""
    propensities = prop_model.predict_proba(X)[:, 1]
    in_overlap = (propensities >= overlap_lo) & (propensities <= overlap_hi)
    frac = in_overlap.mean()
    if frac < overlap_min_fraction:
        return False, f"OptionB: overlap_fraction={frac:.3f} < min_fraction={overlap_min_fraction}"
    return True, None


def check_cdv_retention(
    cdv_df: pd.DataFrame,
    w_cols: list,
    prop_model,
    config: dict,
) -> tuple:
    """
    Apply all Option-C retention criteria to a CDV's training cases.

    Returns (retained: bool, violations: list[str]).
    Violations list is empty if the CDV passes all checks.
    """
    violations = []
    n = len(cdv_df)
    n_treated = int((cdv_df["t"] == 1).sum())
    n_control = int((cdv_df["t"] == 0).sum())

    if n < config["CDV_N_MIN"]:
        violations.append(f"n={n} < CDV_N_MIN={config['CDV_N_MIN']}")

    if n_treated < config["CDV_MIN_ARM_SIZE"]:
        violations.append(
            f"n_treated={n_treated} < CDV_MIN_ARM_SIZE={config['CDV_MIN_ARM_SIZE']}"
        )
    if n_control < config["CDV_MIN_ARM_SIZE"]:
        violations.append(
            f"n_control={n_control} < CDV_MIN_ARM_SIZE={config['CDV_MIN_ARM_SIZE']}"
        )

    ok_a, msg_a = _check_option_a(
        cdv_df["t"].values, config["OVERLAP_LO"], config["OVERLAP_HI"]
    )
    if not ok_a:
        violations.append(msg_a)

    if prop_model is not None and n > 0:
        X_cdv = cdv_df[w_cols].values
        ok_b, msg_b = _check_option_b(
            X_cdv,
            prop_model,
            config["OVERLAP_LO"],
            config["OVERLAP_HI"],
            config["OVERLAP_MIN_FRACTION"],
        )
        if not ok_b:
            violations.append(msg_b)

    return len(violations) == 0, violations


# ============================================================
# CASE ROUTING
# ============================================================

def route_cases(
    df: pd.DataFrame,
    retained_info: dict,
    w_cols: list,
    sentinel_value: float,
) -> pd.Series:
    """
    Assign each row to its CDV ID (e.g. 'CDV_1') or 'OTHER'.

    retained_info: {cdv_id: {'pattern': str, ...}}
    Returns a pd.Series of the same index as df.
    """
    patterns = compute_feature_patterns(df, w_cols, sentinel_value)
    pattern_to_cdv = {info["pattern"]: cdv_id for cdv_id, info in retained_info.items()}
    return patterns.map(lambda p: pattern_to_cdv.get(p, "OTHER"))


# ============================================================
# SUPPORT STATISTICS
# ============================================================

def compute_support_stats(
    df_train: pd.DataFrame,
    cdv_assignment: pd.Series,
    prop_model,
    w_cols: list,
    config: dict,
) -> pd.DataFrame:
    """
    Compute per-CDV support and overlap statistics for one outer seed.

    Returns a DataFrame with one row per CDV group (including OTHER).
    """
    rows = []
    lo = config["OVERLAP_LO"]
    hi = config["OVERLAP_HI"]

    for cdv_id in sorted(cdv_assignment.unique()):
        mask = cdv_assignment == cdv_id
        cdv_df = df_train[mask]
        n = len(cdv_df)
        n_treated = int((cdv_df["t"] == 1).sum())
        n_control = int((cdv_df["t"] == 0).sum())
        treatment_rate = n_treated / n if n > 0 else np.nan

        mean_prop = np.nan
        overlap_frac = np.nan
        pct_outside = np.nan
        ess = np.nan

        if prop_model is not None and n > 0:
            X = cdv_df[w_cols].values
            props = prop_model.predict_proba(X)[:, 1]
            mean_prop = float(props.mean())
            in_overlap = (props >= lo) & (props <= hi)
            overlap_frac = float(in_overlap.mean())
            pct_outside = float((1 - overlap_frac) * 100)
            t_arr = cdv_df["t"].values.flatten()
            weights = np.where(t_arr == 1, 1.0 / np.clip(props, 1e-6, 1 - 1e-6),
                               1.0 / np.clip(1 - props, 1e-6, 1 - 1e-6))
            ess = float((weights.sum() ** 2) / (weights ** 2).sum())

        rows.append({
            "cdv_id": cdv_id,
            "n": n,
            "n_treated": n_treated,
            "n_control": n_control,
            "treatment_rate": treatment_rate,
            "mean_propensity": mean_prop,
            "overlap_fraction": overlap_frac,
            "pct_outside_overlap": pct_outside,
            "ess": ess,
        })

    return pd.DataFrame(rows)


# ============================================================
# RANDOM PARTITION GENERATION
# ============================================================

def generate_random_partition(
    df_train: pd.DataFrame,
    retained_info: dict,
    cdv_assignment_train: pd.Series,
    perm_seed: int,
) -> pd.Series:
    """
    Create a random partition of training cases that exactly preserves each CDV's
    (n_total, n_treated, n_control).

    Treated cases are shuffled among themselves and assigned to random groups.
    Control cases are shuffled independently and assigned to random groups.

    Returns a pd.Series (same index as df_train) with random group labels
    matching CDV IDs (e.g., 'CDV_1', 'CDV_2', ..., 'OTHER').
    """
    rng = np.random.default_rng(perm_seed)

    treated_idx = df_train.index[df_train["t"] == 1].tolist()
    control_idx = df_train.index[df_train["t"] == 0].tolist()

    rng.shuffle(treated_idx)
    rng.shuffle(control_idx)

    assignment = pd.Series("OTHER", index=df_train.index, name="random_cdv")
    treated_ptr, control_ptr = 0, 0

    for cdv_id, info in retained_info.items():
        n_treated = int(info["n_treated"])
        n_control = int(info["n_control"])

        for idx in treated_idx[treated_ptr: treated_ptr + n_treated]:
            assignment.at[idx] = cdv_id
        treated_ptr += n_treated

        for idx in control_idx[control_ptr: control_ptr + n_control]:
            assignment.at[idx] = cdv_id
        control_ptr += n_control

    # Assertions: exact preservation of group sizes and arm counts
    for cdv_id, info in retained_info.items():
        grp = df_train[assignment == cdv_id]
        assert len(grp) == info["n"], f"Size mismatch for {cdv_id}: {len(grp)} != {info['n']}"
        assert int((grp["t"] == 1).sum()) == info["n_treated"], f"Treated mismatch for {cdv_id}"
        assert int((grp["t"] == 0).sum()) == info["n_control"], f"Control mismatch for {cdv_id}"

    return assignment

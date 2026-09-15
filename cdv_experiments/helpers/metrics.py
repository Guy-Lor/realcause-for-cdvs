"""
Metrics computation for the revised CDV experiment.

Primary metrics: ATE MSE, CATE MSE.
Ranking metrics: Kendall tau, Spearman rho.
Paired confidence intervals using outer seeds as independent units.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kendalltau, spearmanr


# ============================================================
# POINT METRICS
# ============================================================

def ate_mse(ite_pred: np.ndarray, ite_true: np.ndarray) -> float:
    """ATE MSE = (mean(ite_pred) - mean(ite_true))^2."""
    return float((np.nanmean(ite_pred) - np.nanmean(ite_true)) ** 2)


def cate_mse(ite_pred: np.ndarray, ite_true: np.ndarray) -> float:
    """CATE MSE = mean((ite_pred - ite_true)^2), ignoring NaN pairs."""
    mask = np.isfinite(ite_pred) & np.isfinite(ite_true)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean((ite_pred[mask] - ite_true[mask]) ** 2))


def ranking_metrics(ite_pred: np.ndarray, ite_true: np.ndarray) -> dict:
    """
    Compute Kendall tau and Spearman rho between predicted and true treatment effects.
    Returns NaN values when ranking is not identifiable (constant true ITE, etc.).
    """
    mask = np.isfinite(ite_pred) & np.isfinite(ite_true)
    ip = ite_pred[mask]
    it = ite_true[mask]

    if len(ip) < 3 or np.std(it) < 1e-8 or np.std(ip) < 1e-8:
        return {"kendall_tau": np.nan, "spearman_rho": np.nan, "rank_defined": False}

    tau, _ = kendalltau(it, ip)
    rho, _ = spearmanr(it, ip)
    return {"kendall_tau": float(tau), "spearman_rho": float(rho), "rank_defined": True}


def ranking_metrics_within_groups(
    ite_pred: np.ndarray, ite_true: np.ndarray, groups: np.ndarray
) -> dict:
    """
    Compute Kendall tau / Spearman rho separately WITHIN each group (e.g. CDV
    assignment, including 'OTHER'), then return the size-weighted average across
    groups. This isolates within-subgroup ranking quality from the pooled/global
    ranking metric (which mixes across-group and within-group ordering).

    Groups with an undefined ranking (rank_defined=False; e.g. too few cases or
    constant true ITE) are skipped and excluded from the weighted average.

    Returns
    -------
    dict with keys: kendall_tau_within, spearman_rho_within, n_groups
        (n_groups = number of groups that contributed a defined ranking).
    """
    groups = np.asarray(groups)
    taus, rhos, weights = [], [], []

    for g in np.unique(groups):
        mask = groups == g
        m = ranking_metrics(ite_pred[mask], ite_true[mask])
        if m["rank_defined"]:
            taus.append(m["kendall_tau"])
            rhos.append(m["spearman_rho"])
            weights.append(int(mask.sum()))

    if not taus:
        return {"kendall_tau_within": np.nan, "spearman_rho_within": np.nan, "n_groups": 0}

    weights = np.array(weights, dtype=float)
    return {
        "kendall_tau_within": float(np.average(taus, weights=weights)),
        "spearman_rho_within": float(np.average(rhos, weights=weights)),
        "n_groups": len(taus),
    }


def all_metrics(ite_pred: np.ndarray, ite_true: np.ndarray) -> dict:
    """Compute all metrics for a single (pred, true) pair."""
    m = {
        "ate_mse": ate_mse(ite_pred, ite_true),
        "cate_mse": cate_mse(ite_pred, ite_true),
        "n": int(np.sum(np.isfinite(ite_pred) & np.isfinite(ite_true))),
    }
    m.update(ranking_metrics(ite_pred, ite_true))
    return m


# ============================================================
# PAIRED CONFIDENCE INTERVALS (across outer seeds)
#
# Every CI/p-value in this module is computed on the per-seed PAIRED
# difference Δ = a - b (same outer seed used for both `a` and `b`, so
# seed-level noise common to both methods cancels out). There is no
# "raw"/unpaired CI reported anywhere: any mean/std shown alongside these is
# purely descriptive (not an inferential interval).
# ============================================================

def resolve_alternative(sided: str, lower_is_better: bool) -> str:
    """
    Map a user-facing 'one-sided'/'two-sided' choice to scipy's `alternative`
    keyword ('two-sided'/'less'/'greater'). One-sided always tests "the
    candidate is better than the reference" in the direction implied by
    `lower_is_better` (True for MSE-type metrics, False for rank-correlation
    metrics where higher is better).
    """
    if sided == "two-sided":
        return "two-sided"
    if sided == "one-sided":
        return "less" if lower_is_better else "greater"
    raise ValueError("sided must be 'one-sided' or 'two-sided'")


def describe_paired_test(
    candidate: str,
    reference: str,
    metric_label: str,
    lower_is_better: bool,
    sided: str,
    ci_method: str = "t",
    alpha: float = 0.05,
) -> str:
    """One-line, human-readable description of a paired CI/test, meant to be printed above its table."""
    test_str = "paired t-test/CI" if ci_method == "t" else "paired bootstrap CI (10,000 resamples)"
    if sided == "two-sided":
        h1 = "Δ ≠ 0"
    else:
        better = "lower" if lower_is_better else "higher"
        sign = "<" if lower_is_better else ">"
        h1 = f"Δ {sign} 0  (i.e. {candidate} has {better} {metric_label} than {reference})"
    return (
        f"{test_str}, {sided} (alpha={alpha:g}). Δ = {candidate} − {reference}  ({metric_label}), "
        f"paired per outer seed. H1: {h1}."
    )


def paired_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    alpha: float = 0.05,
    method: str = "t",
    alternative: str = "two-sided",
) -> dict:
    """
    Paired CI/test on the per-seed difference Δ = a - b.

    Parameters
    ----------
    values_a, values_b : array-like, shape (n_seeds,)
        Per-seed metric values to compare (paired by position/seed).
    alpha : float
        Significance level (e.g. 0.05 -> 95% CI).
    method : str
        'bootstrap' or 't' for a paired t-based CI/test on the deltas.
    alternative : str
        'two-sided' -> symmetric CI, H1: Δ ≠ 0.
        'less'      -> one-sided CI (-inf, upper], H1: Δ < 0.
        'greater'   -> one-sided CI [lower, inf), H1: Δ > 0.

    Returns
    -------
    dict with keys: mean, ci_lo, ci_hi, n, p_value, alternative, test_description
    (ci_lo/ci_hi and p_value are always derived from the same `deltas` array,
    i.e. the CI and the p-value are two views of the same test, not independent).
    """
    if alternative not in ("two-sided", "less", "greater"):
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

    deltas = np.asarray(values_a, dtype=float) - np.asarray(values_b, dtype=float)
    deltas = deltas[np.isfinite(deltas)]
    n = len(deltas)
    mean_d = float(np.mean(deltas)) if n else np.nan
    std_d = float(np.std(deltas, ddof=1)) if n > 1 else np.nan
    test_str = "paired t-test/CI" if method == "t" else "paired bootstrap (10,000 resamples)"
    h1 = {"two-sided": "Δ ≠ 0", "less": "Δ < 0", "greater": "Δ > 0"}[alternative]
    test_description = f"{test_str}, {alternative}, n={n}, H1: {h1}"

    if n < 2:
        return {"mean": mean_d, "std": std_d, "ci_lo": np.nan, "ci_hi": np.nan, "n": n, "p_value": np.nan,
                "alternative": alternative, "test_description": test_description}

    if method == "bootstrap":
        rng = np.random.default_rng(0)
        boot = np.array([np.mean(rng.choice(deltas, size=n, replace=True)) for _ in range(10_000)])
        if alternative == "two-sided":
            ci_lo = float(np.percentile(boot, 100 * alpha / 2))
            ci_hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
            p_value = float(2 * min(np.mean(boot >= 0), np.mean(boot <= 0)))
        elif alternative == "less":
            ci_lo, ci_hi = -np.inf, float(np.percentile(boot, 100 * (1 - alpha)))
            p_value = float(np.mean(boot >= 0))
        else:  # greater
            ci_lo, ci_hi = float(np.percentile(boot, 100 * alpha)), np.inf
            p_value = float(np.mean(boot <= 0))
        p_value = float(min(p_value, 1.0))
    else:  # paired t
        se = float(stats.sem(deltas))
        if se == 0:
            ci_lo = ci_hi = mean_d
            p_value = 1.0 if mean_d == 0 else 0.0
        else:
            if alternative == "two-sided":
                t_crit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
                ci_lo, ci_hi = mean_d - t_crit * se, mean_d + t_crit * se
            elif alternative == "less":
                t_crit = float(stats.t.ppf(1 - alpha, df=n - 1))
                ci_lo, ci_hi = -np.inf, mean_d + t_crit * se
            else:  # greater
                t_crit = float(stats.t.ppf(1 - alpha, df=n - 1))
                ci_lo, ci_hi = mean_d - t_crit * se, np.inf
            _, p_value = stats.ttest_1samp(deltas, 0, alternative=alternative)
            p_value = float(p_value)

    return {"mean": mean_d, "std": std_d, "ci_lo": float(ci_lo), "ci_hi": float(ci_hi), "n": n,
            "p_value": p_value, "alternative": alternative, "test_description": test_description}


def format_ci(lo: float, hi: float, decimals: int = 4) -> str:
    """Format a CI as '[lo, hi]'. One-sided infinite bounds render as '-inf'/'inf'; missing as 'nan'."""
    def fmt(x):
        if np.isnan(x):
            return "nan"
        if np.isinf(x):
            return "-inf" if x < 0 else "inf"
        return f"{x:.{decimals}f}"
    return f"[{fmt(lo)}, {fmt(hi)}]"


# ============================================================
# SUMMARY TABLE BUILDERS
# ============================================================

METHODS_ORDER = [
    "GLOBAL_SENTINEL",
    "GLOBAL_MISSINGNESS",
    "GLOBAL_MISSINGNESS_CDV",
    "MATCHED_RANDOM_PARTITIONS",
    "CDV_SEPARATE",
]

LEARNERS_ORDER = ["DR_RF", "S_RF", "S_Linear", "T_RF", "X_RF", "Double_ML"]


def build_primary_table(
    results_by_seed: dict,
    learner: str = "DR_RF",
    baseline: str = "GLOBAL_SENTINEL",
    ci_method: str = "t",
    sided: str = "one-sided",
) -> pd.DataFrame:
    """
    Build the primary results table (DR_RF only by default).

    Rows: methods. Columns: raw ATE/CATE MSE mean ± std (descriptive only —
    NOT a confidence interval, no inference attached), plus the PAIRED delta
    vs `baseline` (Δ = method − baseline; negative ⇒ method has lower/better
    MSE than baseline) with its paired CI and p-value from `paired_ci`
    (see `resolve_alternative` for what `sided` implies). The baseline's own
    row has no delta (it is the reference against which everything else is
    compared).
    """
    alt = resolve_alternative(sided, lower_is_better=True)

    per_seed = {m: {"ate_mse": [], "cate_mse": []} for m in METHODS_ORDER}
    for seed_result in results_by_seed.values():
        for method in METHODS_ORDER:
            m_data = seed_result.get("metrics", {}).get(method, {}).get(learner, {})
            per_seed[method]["ate_mse"].append(m_data.get("ate_mse", np.nan))
            per_seed[method]["cate_mse"].append(m_data.get("cate_mse", np.nan))

    baseline_ate = np.array(per_seed[baseline]["ate_mse"])
    baseline_cate = np.array(per_seed[baseline]["cate_mse"])

    rows = []
    for method in METHODS_ORDER:
        ate_vals = np.array(per_seed[method]["ate_mse"])
        cate_vals = np.array(per_seed[method]["cate_mse"])

        row = {
            "Method": method,
            "ATE MSE (mean)": float(np.nanmean(ate_vals)),
            "ATE MSE (std)": float(np.nanstd(ate_vals)),
            "CATE MSE (mean)": float(np.nanmean(cate_vals)),
            "CATE MSE (std)": float(np.nanstd(cate_vals)),
            "N seeds": int(np.sum(np.isfinite(ate_vals))),
        }

        if method == baseline:
            row[f"Δ ATE MSE (method − {baseline}) [paired mean]"] = "-"
            row["Δ ATE MSE paired CI"] = "-"
            row["Δ ATE MSE p-value (paired)"] = "-"
            row[f"Δ CATE MSE (method − {baseline}) [paired mean]"] = "-"
            row["Δ CATE MSE paired CI"] = "-"
            row["Δ CATE MSE p-value (paired)"] = "-"
        else:
            d_ate = paired_ci(ate_vals, baseline_ate, method=ci_method, alternative=alt)
            d_cate = paired_ci(cate_vals, baseline_cate, method=ci_method, alternative=alt)
            row[f"Δ ATE MSE (method − {baseline}) [paired mean]"] = d_ate["mean"]
            row["Δ ATE MSE paired CI"] = format_ci(d_ate["ci_lo"], d_ate["ci_hi"])
            row["Δ ATE MSE p-value (paired)"] = d_ate["p_value"]
            row[f"Δ CATE MSE (method − {baseline}) [paired mean]"] = d_cate["mean"]
            row["Δ CATE MSE paired CI"] = format_ci(d_cate["ci_lo"], d_cate["ci_hi"])
            row["Δ CATE MSE p-value (paired)"] = d_cate["p_value"]

        rows.append(row)

    return pd.DataFrame(rows)


def build_full_learner_table(results_by_seed: dict, metric: str = "ate_mse") -> pd.DataFrame:
    """
    Build a method × learner table with mean metric values across seeds.
    """
    data = {}
    for method in METHODS_ORDER:
        data[method] = {}
        for learner in LEARNERS_ORDER:
            vals = [
                sr.get("metrics", {}).get(method, {}).get(learner, {}).get(metric, np.nan)
                for sr in results_by_seed.values()
            ]
            data[method][learner] = float(np.nanmean(vals))

    return pd.DataFrame(data, index=LEARNERS_ORDER).T


def ranking_comparison_table(
    results_by_seed: dict,
    learner: str,
    metric: str = "spearman_rho",
    ci_method: str = "t",
    sided: str = "one-sided",
) -> pd.DataFrame:
    """
    Paired comparison of CDV_SEPARATE vs each other method on a ranking metric
    (higher is better), for one learner. Δ = CDV_SEPARATE − method (positive
    ⇒ CDV_SEPARATE ranks better); see `resolve_alternative` for what `sided`
    implies about the p-value's H1.
    Index: method. Columns: mean, std (raw per-seed values of `method`,
    descriptive only), the paired 95% CI of Δ, p-value (paired, same Δ as
    the CI), and improvement_pct (from seed-mean values, descriptive only).
    """
    alt = resolve_alternative(sided, lower_is_better=False)

    cdv_vals = np.array([
        sr.get("metrics", {}).get("CDV_SEPARATE", {}).get(learner, {}).get(metric, np.nan)
        for sr in results_by_seed.values()
    ])
    cdv_mean = float(np.nanmean(cdv_vals))

    rows = []
    for method in METHODS_ORDER:
        if method == "CDV_SEPARATE":
            continue
        other_vals = np.array([
            sr.get("metrics", {}).get(method, {}).get(learner, {}).get(metric, np.nan)
            for sr in results_by_seed.values()
        ])
        other_mean = float(np.nanmean(other_vals))
        ci = paired_ci(cdv_vals, other_vals, method=ci_method, alternative=alt)
        improvement_pct = (cdv_mean - other_mean) / abs(other_mean) * 100 if other_mean != 0 else np.nan
        rows.append({
            "method": method,
            "mean": other_mean,
            "std": float(np.nanstd(other_vals)),
            "paired 95% CI of Δ (CDV_SEPARATE − method)": format_ci(ci["ci_lo"], ci["ci_hi"]),
            "p_value (paired)": ci["p_value"],
            "improvement_pct": improvement_pct,
        })

    return pd.DataFrame(rows).set_index("method")


# ============================================================
# PAPER-READY SUMMARY TABLES
# ============================================================

def significance_stars(p_value: float) -> str:
    """Map a paired p-value to stars: p<0.01 '***', p<0.05 '**', p<0.1 '*', else 'ns' (not significant)."""
    if p_value is None or not np.isfinite(p_value):
        return "ns"
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.1:
        return "*"
    return "ns"


def adjust_pvalues(p_values: list, method: str = "fdr_bh") -> list:
    """
    Adjust raw p-values for multiplicity across many comparisons.

    method: 'fdr_bh' (Benjamini-Hochberg, controls false discovery rate) or
    'holm' (Holm-Bonferroni step-down, controls family-wise error rate).
    NaN entries are ignored by the adjustment and returned as NaN.
    """
    if method not in ("fdr_bh", "holm"):
        raise ValueError("method must be 'fdr_bh' or 'holm'")

    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(p.shape, np.nan)
    idx = np.where(np.isfinite(p))[0]
    m = len(idx)
    if m == 0:
        return adjusted.tolist()

    order = idx[np.argsort(p[idx])]
    sorted_p = p[order]

    if method == "holm":
        adj_sorted = np.empty(m)
        running_max = 0.0
        for i in range(m):
            running_max = max(running_max, (m - i) * sorted_p[i])
            adj_sorted[i] = min(running_max, 1.0)
    else:  # fdr_bh
        adj_sorted = np.empty(m)
        running_min = 1.0
        for i in range(m - 1, -1, -1):
            running_min = min(running_min, sorted_p[i] * m / (i + 1))
            adj_sorted[i] = min(running_min, 1.0)

    adjusted[order] = adj_sorted
    return adjusted.tolist()


def _adjust_pvalues_by_family(entries: list, method: str = None) -> dict:
    """
    Adjust p-values for multiplicity separately within each named family.

    entries: list of (key, p_value, family_label) tuples. p-values sharing a
    family_label are corrected together (see `adjust_pvalues`); different
    family_labels never share a correction. Returns {key: adjusted_p_value}
    (raw p_value if `method` is None).
    """
    groups = {}
    for key, p, family in entries:
        groups.setdefault(family, ([], []))
        groups[family][0].append(key)
        groups[family][1].append(p)
    p_used = {}
    for keys, p_values in groups.values():
        adjusted = adjust_pvalues(p_values, method=method) if method else p_values
        p_used.update(zip(keys, adjusted))
    return p_used


def build_paper_summary_table(
    results_by_seed: dict,
    learner: str,
    metrics: list,
    metric_labels: dict = None,
    method_labels: dict = None,
    lower_is_better: dict = None,
    ci_method: str = "t",
    sided: str = "two-sided",
    decimals: int = 4,
    adjust_method: str = None,
) -> pd.DataFrame:
    """
    Paper-ready summary table with a 2-level row index (Metric, Method) and
    columns 'Mean ± Std', 'Paired 95% CI (Δ vs CDV_SEPARATE)', 'Sig.'.
    CDV_SEPARATE is always the first row within each metric block (reference:
    its own raw mean ± std, no CI/significance). Every other method's row
    shows its own raw mean ± std (descriptive) plus the paired Δ = CDV_SEPARATE
    − method CI/significance (see `paired_ci`, `significance_stars`).

    If `adjust_method` is set ('fdr_bh' or 'holm'), 'Sig.' uses p-values
    adjusted for multiplicity across all (metric, method) comparisons in this
    table (see `adjust_pvalues`) instead of the raw per-comparison p-value.
    """
    metric_labels = metric_labels or {m: m for m in metrics}
    method_labels = method_labels or {m: m for m in METHODS_ORDER}
    lower_is_better = lower_is_better if lower_is_better is not None else {m: True for m in metrics}

    rows, index_tuples = [], []
    flat_p, flat_idx = [], []
    for metric in metrics:
        alt = resolve_alternative(sided, lower_is_better=lower_is_better[metric])

        cdv_vals = np.array([
            sr.get("metrics", {}).get("CDV_SEPARATE", {}).get(learner, {}).get(metric, np.nan)
            for sr in results_by_seed.values()
        ])
        index_tuples.append((metric_labels[metric], method_labels.get("CDV_SEPARATE", "CDV_SEPARATE")))
        rows.append({
            "Mean ± Std": f"{np.nanmean(cdv_vals):.{decimals}f} ± {np.nanstd(cdv_vals):.{decimals}f}",
            "Paired 95% CI (Δ vs CDV_SEPARATE)": "-",
            "Sig.": "-",
        })

        for method in METHODS_ORDER:
            if method == "CDV_SEPARATE":
                continue
            other_vals = np.array([
                sr.get("metrics", {}).get(method, {}).get(learner, {}).get(metric, np.nan)
                for sr in results_by_seed.values()
            ])
            ci = paired_ci(cdv_vals, other_vals, method=ci_method, alternative=alt)
            index_tuples.append((metric_labels[metric], method_labels.get(method, method)))
            rows.append({
                "Mean ± Std": f"{np.nanmean(other_vals):.{decimals}f} ± {np.nanstd(other_vals):.{decimals}f}",
                "Paired 95% CI (Δ vs CDV_SEPARATE)": format_ci(ci["ci_lo"], ci["ci_hi"], decimals=decimals),
                "Sig.": ci["p_value"],
            })
            if np.isfinite(ci["p_value"]):
                flat_p.append(ci["p_value"])
                flat_idx.append(len(rows) - 1)

    p_used = dict(zip(flat_idx, adjust_pvalues(flat_p, method=adjust_method) if adjust_method else flat_p))
    for i, row in enumerate(rows):
        if row["Sig."] != "-":
            row["Sig."] = significance_stars(p_used.get(i, row["Sig."]))

    index = pd.MultiIndex.from_tuples(index_tuples, names=["Metric", "Method"])
    return pd.DataFrame(rows, index=index)


def build_paper_summary_table_by_alpha(
    results_by_alpha: dict,
    learner: str,
    alpha_values: list,
    metrics: list,
    metric_labels: dict = None,
    method_labels: dict = None,
    lower_is_better: dict = None,
    ci_method: str = "t",
    sided: str = "two-sided",
    decimals: int = 3,
    adjust_method: str = None,
    metric_families: dict = None,
) -> pd.DataFrame:
    """
    Compact paper-ready summary across heterogeneity levels (alpha): rows are
    (Metric, Method), columns are alpha values. The CDV_SEPARATE row shows its
    own raw "mean ± std" per alpha (descriptive reference). Every other
    method's cell shows the paired Δ = CDV_SEPARATE − method as "mean ± std"
    (mean and std of the per-seed paired differences) with significance stars
    from the same paired test, e.g. "-0.938 ± 0.120***"
    (see `paired_ci`, `significance_stars`). A cell is "-" when no data exists
    for that alpha.

    If `adjust_method` is set ('fdr_bh' or 'holm'), stars use p-values
    adjusted for multiplicity per FDR family (see `_adjust_pvalues_by_family`);
    each alpha is always its own family, never pooled across alpha. Within an
    alpha, `metric_families` maps metric -> family label so metrics sharing a
    label are corrected together; metrics absent from the dict each form
    their own singleton family. If `metric_families` is None, every metric in
    this table shares one family per alpha (previous behavior).
    """
    metric_labels = metric_labels or {m: m for m in metrics}
    method_labels = method_labels or {m: m for m in METHODS_ORDER}
    lower_is_better = lower_is_better if lower_is_better is not None else {m: True for m in metrics}

    # Pass 1: compute every cell (CDV reference rows as strings, other rows as
    # paired_ci dicts keyed by (metric, method, alpha)) and collect p-values.
    index_tuples, cdv_rows = [], []
    cell_ci = {}
    entries = []  # (key, p_value, (alpha, family_label))
    for metric in metrics:
        alt = resolve_alternative(sided, lower_is_better=lower_is_better[metric])

        index_tuples.append((metric_labels[metric], method_labels.get("CDV_SEPARATE", "CDV_SEPARATE")))
        cdv_row = {}
        for alpha in alpha_values:
            res = results_by_alpha.get(alpha, {})
            cdv_vals = np.array([
                sr.get("metrics", {}).get("CDV_SEPARATE", {}).get(learner, {}).get(metric, np.nan)
                for sr in res.values()
            ])
            cdv_row[alpha] = ("-" if np.all(~np.isfinite(cdv_vals))
                               else f"{np.nanmean(cdv_vals):.{decimals}f} ± {np.nanstd(cdv_vals):.{decimals}f}")
        cdv_rows.append(cdv_row)

        for method in METHODS_ORDER:
            if method == "CDV_SEPARATE":
                continue
            index_tuples.append((metric_labels[metric], method_labels.get(method, method)))
            for alpha in alpha_values:
                res = results_by_alpha.get(alpha, {})
                cdv_vals = np.array([
                    sr.get("metrics", {}).get("CDV_SEPARATE", {}).get(learner, {}).get(metric, np.nan)
                    for sr in res.values()
                ])
                other_vals = np.array([
                    sr.get("metrics", {}).get(method, {}).get(learner, {}).get(metric, np.nan)
                    for sr in res.values()
                ])
                ci = paired_ci(cdv_vals, other_vals, method=ci_method, alternative=alt)
                key = (metric, method, alpha)
                cell_ci[key] = ci
                if np.isfinite(ci["mean"]):
                    family_label = metric_families.get(metric, metric) if metric_families is not None else "ALL"
                    entries.append((key, ci["p_value"], (alpha, family_label)))

    p_used = _adjust_pvalues_by_family(entries, method=adjust_method)

    # Pass 2: render rows in the same (metric, method) order as index_tuples.
    rows = []
    cdv_iter = iter(cdv_rows)
    for metric in metrics:
        rows.append(next(cdv_iter))
        for method in METHODS_ORDER:
            if method == "CDV_SEPARATE":
                continue
            row = {}
            for alpha in alpha_values:
                ci = cell_ci[(metric, method, alpha)]
                if not np.isfinite(ci["mean"]):
                    row[alpha] = "-"
                else:
                    std_str = f"{ci['std']:.{decimals}f}" if np.isfinite(ci["std"]) else "nan"
                    stars = significance_stars(p_used[(metric, method, alpha)])
                    row[alpha] = f"{ci['mean']:.{decimals}f} ± {std_str} {stars}"
            rows.append(row)

    index = pd.MultiIndex.from_tuples(index_tuples, names=["Metric", "Method"])
    df = pd.DataFrame(rows, index=index)
    df.columns = [f"α={a:.2f}" for a in alpha_values]
    return df


def _signed_delta_pct(cdv_mean: float, other_mean: float, lower_is_better: bool) -> float:
    """Signed % improvement of CDV_SEPARATE vs other; positive always ⇒ CDV_SEPARATE better."""
    if other_mean == 0 or not np.isfinite(other_mean) or not np.isfinite(cdv_mean):
        return np.nan
    if lower_is_better:
        return (other_mean - cdv_mean) / abs(other_mean) * 100
    return (cdv_mean - other_mean) / abs(other_mean) * 100


def _format_signed_delta(delta_pct: float, p_value: float) -> str:
    """Format a signed % delta with significance stars, e.g. '+12.4%**'."""
    if not np.isfinite(delta_pct):
        return "-"
    sign = "+" if delta_pct >= 0 else ""
    return f"{sign}{delta_pct:.1f}%{significance_stars(p_value)}"


def build_signed_summary_table(
    results_by_seed: dict,
    learner: str,
    metrics: list,
    metric_labels: dict = None,
    method_labels: dict = None,
    lower_is_better: dict = None,
    ci_method: str = "t",
    sided: str = "two-sided",
    decimals: int = 4,
    adjust_method: str = None,
    metric_families: dict = None,
) -> pd.DataFrame:
    """
    Compact single-dataset summary: rows=method (all of `METHODS_ORDER`,
    CDV_SEPARATE first as reference), columns=(Metric, ['Mean ± Std',
    'Δ% (sig.)']). Δ% is SIGNED so that positive always means CDV_SEPARATE is
    better (regardless of whether the metric is lower- or higher-is-better),
    with significance stars from a paired test (`sided`, default two-sided) on
    the raw per-seed values (see `paired_ci`). CDV_SEPARATE's own row shows
    only its raw mean ± std ('—' for Δ%, since it is the reference).

    If `adjust_method` is set ('fdr_bh' or 'holm'), stars use p-values
    adjusted for multiplicity per FDR family (see `_adjust_pvalues_by_family`).
    `metric_families` maps metric -> family label so metrics sharing a label
    are corrected together (e.g. {'kendall_tau': 'RANK', 'spearman_rho':
    'RANK'} pools those two into one family); metrics absent from the dict
    each form their own singleton family. If `metric_families` is None, every
    metric in this table is treated as a single shared family (previous
    behavior).
    """
    metric_labels = metric_labels or {m: m for m in metrics}
    method_labels = method_labels or {m: m for m in METHODS_ORDER}
    lower_is_better = lower_is_better if lower_is_better is not None else {m: True for m in metrics}

    rows, row_labels = [], []
    pending, delta_pct_by_key, p_by_key, entries = [], {}, {}, []
    for method in METHODS_ORDER:
        row_labels.append(method_labels.get(method, method))
        row = {}
        for metric in metrics:
            label = metric_labels[metric]
            vals = np.array([
                sr.get("metrics", {}).get(method, {}).get(learner, {}).get(metric, np.nan)
                for sr in results_by_seed.values()
            ])
            row[(label, "Mean ± Std")] = f"{np.nanmean(vals):.{decimals}f} ± {np.nanstd(vals):.{decimals}f}"
            if method == "CDV_SEPARATE":
                row[(label, "Δ% (sig.)")] = "—"
                continue
            cdv_vals = np.array([
                sr.get("metrics", {}).get("CDV_SEPARATE", {}).get(learner, {}).get(metric, np.nan)
                for sr in results_by_seed.values()
            ])
            alt = resolve_alternative(sided, lower_is_better=lower_is_better[metric])
            ci = paired_ci(cdv_vals, vals, method=ci_method, alternative=alt)
            key = len(pending)
            delta_pct_by_key[key] = _signed_delta_pct(
                float(np.nanmean(cdv_vals)), float(np.nanmean(vals)), lower_is_better[metric])
            p_by_key[key] = ci["p_value"]
            if np.isfinite(ci["p_value"]):
                family = metric_families.get(metric, metric) if metric_families is not None else "ALL"
                entries.append((key, ci["p_value"], family))
            pending.append((row, (label, "Δ% (sig.)"), key))
        rows.append(row)

    p_used = _adjust_pvalues_by_family(entries, method=adjust_method)
    for row, col_delta, key in pending:
        row[col_delta] = _format_signed_delta(delta_pct_by_key[key], p_used.get(key, p_by_key[key]))

    df = pd.DataFrame(rows, index=row_labels)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["Metric", ""])
    return df


def build_signed_summary_table_by_alpha(
    results_by_alpha: dict,
    learner: str,
    alpha_values: list,
    metrics: list,
    metric_labels: dict = None,
    method_labels: dict = None,
    lower_is_better: dict = None,
    ci_method: str = "t",
    sided: str = "two-sided",
    decimals: int = 3,
    adjust_method: str = None,
    metric_families: dict = None,
) -> pd.DataFrame:
    """
    Compact heterogeneity-sweep summary: rows=method (CDV_SEPARATE first,
    reference), columns=(Metric, alpha, ['Mean ± Std', 'Δ% (sig.)']). Same
    signed-Δ% convention as `build_signed_summary_table` (positive ⇒
    CDV_SEPARATE better), stars from a paired test (`sided`, default
    two-sided). A cell is '-' when no data exists for that alpha/metric.

    If `adjust_method` is set ('fdr_bh' or 'holm'), stars use p-values
    adjusted for multiplicity per FDR family (see `_adjust_pvalues_by_family`);
    each alpha is always its own family, never pooled across alpha. Within an
    alpha, `metric_families` maps metric -> family label so metrics sharing a
    label are corrected together; metrics absent from the dict each form
    their own singleton family. If `metric_families` is None, every metric in
    this table shares one family per alpha (previous behavior).
    """
    metric_labels = metric_labels or {m: m for m in metrics}
    method_labels = method_labels or {m: m for m in METHODS_ORDER}
    lower_is_better = lower_is_better if lower_is_better is not None else {m: True for m in metrics}

    # Pass 1: compute every cell's value/delta%/p-value, collect p-values for adjustment.
    rows, row_labels = [], []
    pending, delta_pct_by_key, p_by_key = [], {}, {}
    entries = []  # (key, p_value, (alpha, family_label))
    for method in METHODS_ORDER:
        row_labels.append(method_labels.get(method, method))
        row = {}
        for metric in metrics:
            label = metric_labels[metric]
            for alpha in alpha_values:
                res = results_by_alpha.get(alpha, {})
                vals = np.array([
                    sr.get("metrics", {}).get(method, {}).get(learner, {}).get(metric, np.nan)
                    for sr in res.values()
                ])
                col_val, col_delta = (label, alpha, "Mean ± Std"), (label, alpha, "Δ% (sig.)")
                if np.all(~np.isfinite(vals)):
                    row[col_val] = "-"
                    row[col_delta] = "-"
                    continue
                row[col_val] = f"{np.nanmean(vals):.{decimals}f} ± {np.nanstd(vals):.{decimals}f}"
                if method == "CDV_SEPARATE":
                    row[col_delta] = "—"
                    continue
                cdv_vals = np.array([
                    sr.get("metrics", {}).get("CDV_SEPARATE", {}).get(learner, {}).get(metric, np.nan)
                    for sr in res.values()
                ])
                alt = resolve_alternative(sided, lower_is_better=lower_is_better[metric])
                ci = paired_ci(cdv_vals, vals, method=ci_method, alternative=alt)
                key = len(pending)
                delta_pct_by_key[key] = _signed_delta_pct(
                    float(np.nanmean(cdv_vals)), float(np.nanmean(vals)), lower_is_better[metric])
                p_by_key[key] = ci["p_value"]
                if np.isfinite(ci["p_value"]):
                    family_label = metric_families.get(metric, metric) if metric_families is not None else "ALL"
                    entries.append((key, ci["p_value"], (alpha, family_label)))
                pending.append((row, col_delta, key))
        rows.append(row)

    p_used = _adjust_pvalues_by_family(entries, method=adjust_method)

    # Pass 2: fill in the deferred Δ% cells with the (possibly adjusted) p-value.
    for row, col_delta, key in pending:
        row[col_delta] = _format_signed_delta(delta_pct_by_key[key], p_used.get(key, p_by_key[key]))

    df = pd.DataFrame(rows, index=row_labels)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["Metric", "alpha", ""])
    return df


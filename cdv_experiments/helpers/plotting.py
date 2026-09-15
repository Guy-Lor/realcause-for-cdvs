"""
Plotting utilities for the revised CDV experiment.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

METHOD_COLORS = {
    "GLOBAL_SENTINEL":           "#d62728",
    "GLOBAL_MISSINGNESS":        "#ff7f0e",
    "GLOBAL_MISSINGNESS_CDV":    "#2ca02c",
    "MATCHED_RANDOM_PARTITIONS": "#9467bd",
    "CDV_SEPARATE":              "#1f77b4",
}
METHOD_MARKERS = {
    "GLOBAL_SENTINEL":           "s",
    "GLOBAL_MISSINGNESS":        "^",
    "GLOBAL_MISSINGNESS_CDV":    "D",
    "MATCHED_RANDOM_PARTITIONS": "v",
    "CDV_SEPARATE":              "o",
}
METHOD_LABELS = {
    "GLOBAL_SENTINEL":           "Global (Sentinel)",
    "GLOBAL_MISSINGNESS":        "Global + Missingness",
    "GLOBAL_MISSINGNESS_CDV":    "Global + Miss. + CDV ID",
    "MATCHED_RANDOM_PARTITIONS": "Matched Random Partitions",
    "CDV_SEPARATE":              "CDV Separate (proposed)",
}


def _per_seed_metric(results_by_seed, method, learner, metric):
    vals = []
    for sr in results_by_seed.values():
        v = sr.get("metrics", {}).get(method, {}).get(learner, {}).get(metric, np.nan)
        vals.append(v)
    return np.array(vals)


def plot_primary_bar_table(results_by_seed, learner="DR_RF", metric="ate_mse",
                           title="Primary Results (DR-RF)", figsize=(10, 5),
                           save_path=None):
    """Horizontal bar chart with 95% CI for each method."""
    methods = list(METHOD_LABELS.keys())
    means, lo_errs, hi_errs = [], [], []

    for m in methods:
        vals = _per_seed_metric(results_by_seed, m, learner, metric)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2:
            means.append(np.nan); lo_errs.append(0); hi_errs.append(0)
            continue
        mean = float(np.mean(vals))
        t_crit = float(stats.t.ppf(0.975, df=len(vals) - 1))
        se = float(stats.sem(vals))
        means.append(mean)
        lo_errs.append(mean - (mean - t_crit * se))
        hi_errs.append((mean + t_crit * se) - mean)

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(methods))
    colors = [METHOD_COLORS[m] for m in methods]
    ax.barh(y_pos, means, xerr=[lo_errs, hi_errs], color=colors, alpha=0.8,
            capsize=5, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([METHOD_LABELS[m] for m in methods], fontsize=11)
    ax.set_xlabel(metric.replace("_", " ").upper(), fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_scissors_chart(results_by_alpha, alpha_values, learner="DR_RF",
                        metric="ate_mse", title=None, figsize=(10, 5),
                        save_path=None):
    """
    Scissors chart for synthetic experiment: one line per method, x=alpha.
    Shaded 95% CI band based on outer seeds.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for method in METHOD_LABELS:
        means, lo_bands, hi_bands = [], [], []
        for alpha in alpha_values:
            res = results_by_alpha.get(alpha, {})
            vals = []
            for sr in res.values():
                v = sr.get("metrics", {}).get(method, {}).get(learner, {}).get(metric, np.nan)
                vals.append(v)
            vals = np.array(vals)[np.isfinite(vals) if len(vals) > 0 else []]
            if len(vals) < 2:
                means.append(np.nan); lo_bands.append(np.nan); hi_bands.append(np.nan)
                continue
            mean = float(np.mean(vals))
            t_crit = float(stats.t.ppf(0.975, df=len(vals) - 1))
            se = float(stats.sem(vals))
            means.append(mean)
            lo_bands.append(mean - t_crit * se)
            hi_bands.append(mean + t_crit * se)

        ax.plot(alpha_values, means, color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method], linewidth=2, markersize=8,
                label=METHOD_LABELS[method])
        ax.fill_between(alpha_values, lo_bands, hi_bands,
                        color=METHOD_COLORS[method], alpha=0.15)

    ax.set_xlabel("α (Heterogeneity Level)", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").upper(), fontsize=12)
    ax.set_title(title or f"{metric.upper()} vs α ({learner})", fontsize=13)
    ax.set_xticks(alpha_values)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_scissors_chart_combined(results_by_alpha, panels, learner="DR_RF",
                                  suptitle=None, figsize=(10, 4), save_path=None):
    """
    Side-by-side scissors charts (one axes per panel) sharing a single legend.

    panels: list of dicts, each with keys:
        - 'metric': metric name (e.g. 'cate_mse', 'kendall_tau')
        - 'alpha_values': list of alpha values to plot for this panel
        - 'title': subplot title
    All methods use the same marker (circle); only color distinguishes them.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=figsize)
    if len(panels) == 1:
        axes = [axes]

    for ax, panel in zip(axes, panels):
        metric = panel["metric"]
        alpha_values = panel["alpha_values"]
        for method in METHOD_LABELS:
            means, lo_bands, hi_bands = [], [], []
            for alpha in alpha_values:
                res = results_by_alpha.get(alpha, {})
                vals = []
                for sr in res.values():
                    v = sr.get("metrics", {}).get(method, {}).get(learner, {}).get(metric, np.nan)
                    vals.append(v)
                vals = np.array(vals)[np.isfinite(vals) if len(vals) > 0 else []]
                if len(vals) < 2:
                    means.append(np.nan); lo_bands.append(np.nan); hi_bands.append(np.nan)
                    continue
                mean = float(np.mean(vals))
                t_crit = float(stats.t.ppf(0.975, df=len(vals) - 1))
                se = float(stats.sem(vals))
                means.append(mean)
                lo_bands.append(mean - t_crit * se)
                hi_bands.append(mean + t_crit * se)

            ax.plot(alpha_values, means, color=METHOD_COLORS[method],
                    marker="o", linewidth=2, markersize=8,
                    label=METHOD_LABELS[method])
            ax.fill_between(alpha_values, lo_bands, hi_bands,
                            color=METHOD_COLORS[method], alpha=0.15)

        ax.set_xlabel("α (Heterogeneity Level)", fontsize=14)
        ax.set_ylabel(metric.replace("_", " ").upper().replace("TAU", "τ"), fontsize=14)
        ax.set_title(panel.get("title") or f"{metric.upper()} vs α ({learner})", fontsize=14)
        ax.set_xticks(alpha_values)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=12, loc="lower center",
               ncol=len(labels), bbox_to_anchor=(0.5, -0.08))
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_cdv_support_heatmap(support_stats_by_seed, title="CDV Support Across Seeds",
                              figsize=(10, 5), save_path=None):
    """
    Heatmap of CDV sample sizes across outer seeds.
    """
    # Build matrix: rows=CDV IDs, cols=seeds
    all_cdvs = set()
    for ss in support_stats_by_seed.values():
        if isinstance(ss, pd.DataFrame):
            all_cdvs.update(ss["cdv_id"].tolist())
    all_cdvs = sorted(all_cdvs)
    seeds = sorted(support_stats_by_seed.keys())

    matrix = np.full((len(all_cdvs), len(seeds)), np.nan)
    for j, seed in enumerate(seeds):
        ss = support_stats_by_seed[seed]
        if not isinstance(ss, pd.DataFrame):
            continue
        for i, cdv in enumerate(all_cdvs):
            row = ss[ss["cdv_id"] == cdv]
            if not row.empty:
                matrix[i, j] = float(row["n"].iloc[0])

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(seeds)))
    ax.set_xticklabels([str(s) for s in seeds], fontsize=7, rotation=90)
    ax.set_yticks(np.arange(len(all_cdvs)))
    ax.set_yticklabels(all_cdvs, fontsize=9)
    ax.set_xlabel("Outer Seed", fontsize=11)
    ax.set_ylabel("CDV", fontsize=11)
    ax.set_title(title, fontsize=12)
    plt.colorbar(im, ax=ax, label="n cases")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

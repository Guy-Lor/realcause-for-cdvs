"""
One-off script (not part of the paper pipeline) to build full method x learner
grids with paired 95% CI vs CDV_SEPARATE, for the paper appendix.
Run with the ICPM_paper2025_2nd_env interpreter from the repo root:
    python cdv_experiments/helpers/appendix_grids.py
"""
import os
import sys
import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "cdv_experiments"))

from helpers.runner import load_checkpoint
from helpers.metrics import (
    METHODS_ORDER, paired_ci, resolve_alternative, significance_stars,
)

LEARNERS = ["DR_RF", "S_RF", "S_Linear", "T_RF", "X_RF"]  # Double_ML excluded (unstable, unused in paper)
METHOD_LABELS = {
    "GLOBAL_SENTINEL": "Global (Sentinel)",
    "GLOBAL_MISSINGNESS": "Global + Missingness",
    "GLOBAL_MISSINGNESS_CDV": "Global + Miss. + CDV ID",
    "MATCHED_RANDOM_PARTITIONS": "Matched Random Partitions",
    "CDV_SEPARATE": "CDV Separate (proposed)",
}
CI_METHOD = "bootstrap"
SIDED = "two-sided"


def build_grid(results_by_seed: dict, metric: str) -> pd.DataFrame:
    alt = resolve_alternative(SIDED, lower_is_better=True)
    cdv_vals_by_learner = {
        learner: np.array([
            sr["metrics"].get("CDV_SEPARATE", {}).get(learner, {}).get(metric, np.nan)
            for sr in results_by_seed.values()
        ])
        for learner in LEARNERS
    }
    rows = []
    for method in METHODS_ORDER:
        row = {"Method": METHOD_LABELS[method]}
        for learner in LEARNERS:
            vals = np.array([
                sr["metrics"].get(method, {}).get(learner, {}).get(metric, np.nan)
                for sr in results_by_seed.values()
            ])
            mean, std = np.nanmean(vals), np.nanstd(vals)
            if method == "CDV_SEPARATE":
                cell = f"{mean:.4g} $\\pm$ {std:.4g}"
            else:
                ci = paired_ci(cdv_vals_by_learner[learner], vals, method=CI_METHOD, alternative=alt)
                stars = significance_stars(ci["p_value"])
                cell = (f"{mean:.4g} $\\pm$ {std:.4g} "
                        f"[{ci['ci_lo']:.3g}, {ci['ci_hi']:.3g}]{stars}")
            row[learner] = cell
        rows.append(row)
    return pd.DataFrame(rows).set_index("Method")


def main():
    out_dir_sepsis = os.path.join(_ROOT, "cdv_experiments", "sepsis", "artifacts")
    out_dir_synth = os.path.join(_ROOT, "cdv_experiments", "synthetic", "artifacts")

    # --- Sepsis ---
    sepsis_res = load_checkpoint(os.path.join(out_dir_sepsis, "results_checkpoint.pkl"))
    for metric in ["ate_mse", "cate_mse"]:
        grid = build_grid(sepsis_res, metric)
        path = os.path.join(out_dir_sepsis, f"appendix_full_grid_{metric}.csv")
        grid.to_csv(path)
        print(f"Saved {path}")
        print(grid.to_string())
        print()

    # --- Synthetic (per alpha) ---
    alpha_values = [0.0, 0.25, 0.50, 0.75, 1.0]
    for alpha in alpha_values:
        path_pkl = os.path.join(out_dir_synth, f"results_alpha_{alpha:.2f}.pkl")
        if not os.path.exists(path_pkl):
            print(f"MISSING: {path_pkl}")
            continue
        res = load_checkpoint(path_pkl)
        print(f"alpha={alpha:.2f}: {len(res)} seeds")
        for metric in ["ate_mse", "cate_mse"]:
            grid = build_grid(res, metric)
            path = os.path.join(out_dir_synth, f"appendix_full_grid_{metric}_alpha_{alpha:.2f}.csv")
            grid.to_csv(path)
            print(f"Saved {path}")
            print(grid.to_string())
            print()


if __name__ == "__main__":
    main()

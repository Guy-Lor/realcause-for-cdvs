"""
Results Analysis Module for Causal Analysis

This module provides functions for analyzing causal inference results including:
- ATE and CATE MSE decomposition (bias-variance analysis)
- Statistical significance testing
- Comprehensive results table generation
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_rel
import warnings
warnings.filterwarnings('ignore')


def identify_fallback_variants(df):
    """Identify variants where variant model equals global model (fallback cases)"""
    df = df.copy()
    
    # Check if there's a column indicating fallback status
    if 'used_global_model' in df.columns:
        df['is_fallback'] = df['used_global_model']
        fallback_count = df['is_fallback'].sum()
        print(f"✓ Using existing 'used_global_model' column: {fallback_count} fallback cases identified")
    else:
        # Alternative: check if estimator is marked as 'global' or contains 'global'
        if 'estimator' in df.columns:
            df['is_fallback'] = df['estimator'].str.contains('global', case=False, na=False)
            fallback_count = df['is_fallback'].sum()
            print(f"✓ Identified {fallback_count} fallback cases based on estimator names")
        else:
            # No way to identify fallbacks, mark all as non-fallback
            df['is_fallback'] = False
            print("⚠️  Could not identify fallback variants, marking all as non-fallback")
    
    return df


def add_error_columns(df):
    """Add error calculation columns"""
    df = df.copy()
    df['ite_error'] = df['ite_pred'] - df['ite_real']
    df['ite_squared_error'] = df['ite_error'] ** 2
    df['ite_abs_error'] = np.abs(df['ite_error'])
    return df


def calculate_ate_mse_decomposition(df_global, df_variant, seeds):
    """
    Calculate ATE-level MSE decomposition comparing best global vs best variant methods.
    
    For each seed:
    1. Compute true ATE = mean(true ITE) over all test samples
    2. Compute estimated ATE = mean(predicted ITE) over all samples
    3. Compute ATE error = (estimated ATE - true ATE)
    
    Across seeds:
    4. Bias = mean of ATE errors across seeds
    5. Variance = variance of ATE errors across seeds  
    6. MSE = mean of squared ATE errors across seeds
    """
    print("Calculating ATE-level MSE decomposition...")
    print("=" * 50)
    
    # Calculate ATE by seed for global method
    global_ate_by_seed = []
    for seed in seeds:
        seed_data = df_global[df_global['seed'] == seed]
        if len(seed_data) > 0:
            true_ate = seed_data['ite_real'].mean()
            estimated_ate = seed_data['ite_pred'].mean()
            ate_error = estimated_ate - true_ate
            
            global_ate_by_seed.append({
                'seed': seed,
                'method': 'global',
                'true_ate': true_ate,
                'estimated_ate': estimated_ate,
                'ate_error': ate_error,
                'ate_squared_error': ate_error ** 2,
                'n_samples': len(seed_data)
            })
    
    # Calculate ATE by seed for variant method
    variant_ate_by_seed = []
    for seed in seeds:
        seed_data = df_variant[df_variant['seed'] == seed]
        if len(seed_data) > 0:
            true_ate = seed_data['ite_real'].mean()
            estimated_ate = seed_data['ite_pred'].mean()
            ate_error = estimated_ate - true_ate
            
            variant_ate_by_seed.append({
                'seed': seed,
                'method': 'variant',
                'true_ate': true_ate,
                'estimated_ate': estimated_ate,
                'ate_error': ate_error,
                'ate_squared_error': ate_error ** 2,
                'n_samples': len(seed_data)
            })
    
    # Combine results
    all_ate_results = pd.DataFrame(global_ate_by_seed + variant_ate_by_seed)
    
    # Calculate bias, variance, MSE for each method
    decomposition_results = []
    
    for method in ['global', 'variant']:
        method_data = all_ate_results[all_ate_results['method'] == method]
        ate_errors = method_data['ate_error'].values
        
        if len(ate_errors) > 0:
            bias = np.mean(ate_errors)
            variance = np.var(ate_errors, ddof=0) if len(ate_errors) > 1 else 0
            mse = np.mean(method_data['ate_squared_error'])
            
            # Verify: MSE should equal bias² + variance (approximately)
            bias_squared = bias ** 2
            theoretical_mse = bias_squared + variance
            
            decomposition_results.append({
                'method': method,
                'bias': bias,
                'variance': variance,
                'bias_squared': bias_squared,
                'mse': mse,
                'mse_theoretical': theoretical_mse,
                'mse_difference': abs(mse - theoretical_mse),
                'n_seeds': len(ate_errors),
                'mean_true_ate': method_data['true_ate'].mean(),
                'mean_estimated_ate': method_data['estimated_ate'].mean()
            })
    
    decomposition_df = pd.DataFrame(decomposition_results)
    
    # Display results
    print("ATE MSE Decomposition Results:")
    print("-" * 50)
    print(decomposition_df[['method', 'bias', 'variance', 'bias_squared', 'mse', 'n_seeds']])
    
    print(f"\nVerification (MSE should ≈ Bias² + Variance):")
    for _, row in decomposition_df.iterrows():
        print(f"{row['method']:>8}: MSE={row['mse']:.6f}, Bias²+Var={row['mse_theoretical']:.6f}, "
              f"Diff={row['mse_difference']:.8f}")
    
    return decomposition_df, all_ate_results


def calculate_ate_by_estimator(df_global, df_variant, seeds):
    """
    Calculate ATE for each estimator using all samples per seed and estimator together.
    Ignores 'is_fallback' flag and computes ATE across all samples for each estimator/seed combination.
    
    For each estimator and seed:
    1. Compute true ATE = mean(true ITE) over all test samples
    2. Compute estimated ATE = mean(predicted ITE) over all samples
    3. Compute ATE error = (estimated ATE - true ATE)
    """
    print("Calculating ATE for each estimator...")
    print("=" * 50)
    
    # Get unique estimators from both datasets
    global_estimators = set(df_global['estimator'].unique()) if 'estimator' in df_global.columns else set()
    variant_estimators = set(df_variant['estimator'].unique()) if 'estimator' in df_variant.columns else set()
    all_estimators = sorted(global_estimators.union(variant_estimators))
    
    print(f"Found {len(all_estimators)} unique estimators: {all_estimators}")
    print()
    
    # ---------- FAST PART: pre-aggregate by (estimator, seed) ----------
    def aggregate_ate(df, method_name):
        if df is None or df.empty:
            return pd.DataFrame(columns=[
                'estimator', 'seed', 'method',
                'true_ate', 'estimated_ate', 'ate_error',
                'ate_squared_error', 'n_samples'
            ])
        # group by estimator and seed, compute means and counts
        grouped = (
            df.groupby(['estimator', 'seed'], as_index=False)
              .agg(
                  true_ate=('ite_real', 'mean'),
                  estimated_ate=('ite_pred', 'mean'),
                  n_samples=('ite_real', 'size')
              )
        )
        grouped['method'] = method_name
        grouped['ate_error'] = grouped['estimated_ate'] - grouped['true_ate']
        grouped['ate_squared_error'] = grouped['ate_error'] ** 2
        return grouped[
            [
                'estimator', 'seed', 'method',
                'true_ate', 'estimated_ate', 'ate_error',
                'ate_squared_error', 'n_samples'
            ]
        ]
    
    global_ate_results = aggregate_ate(df_global, 'global')
    variant_ate_results = aggregate_ate(df_variant, 'variant')
    
    # Combine all results (same structure as before)
    all_ate_results = pd.concat([global_ate_results, variant_ate_results], ignore_index=True)
    
    # ---------- Summary statistics by estimator and method ----------
    estimator_summary_results = []
    
    for estimator in all_estimators:
        estimator_data = all_ate_results[all_ate_results['estimator'] == estimator]
        
        for method in ['global', 'variant']:
            method_data = estimator_data[estimator_data['method'] == method]
            
            if len(method_data) > 0:
                ate_errors = method_data['ate_error'].values
                
                bias = np.mean(ate_errors)
                variance = np.var(ate_errors, ddof=0) if len(ate_errors) > 1 else 0
                mse = np.mean(method_data['ate_squared_error'])
                
                # Verify: MSE should equal bias² + variance (approximately)
                bias_squared = bias ** 2
                theoretical_mse = bias_squared + variance
                
                estimator_summary_results.append({
                    'estimator': estimator,
                    'method': method,
                    'bias': bias,
                    'variance': variance,
                    'bias_squared': bias_squared,
                    'mse': mse,
                    'mse_theoretical': theoretical_mse,
                    'mse_difference': abs(mse - theoretical_mse),
                    'n_seeds': len(ate_errors),
                    'mean_true_ate': method_data['true_ate'].mean(),
                    'mean_estimated_ate': method_data['estimated_ate'].mean(),
                    'total_samples': method_data['n_samples'].sum()
                })
    
    estimator_summary_df = pd.DataFrame(estimator_summary_results)
    
    # Create pivot table for easier comparison
    comparison_df = estimator_summary_df.pivot_table(
        index='estimator',
        columns='method',
        values=['bias', 'variance', 'mse'],
        fill_value=np.nan
    )
    
    print("\nMSE Comparison (Global vs Variant):")
    if 'mse' in comparison_df.columns:
        mse_comparison = comparison_df['mse'].copy()
        if 'global' in mse_comparison.columns and 'variant' in mse_comparison.columns:
            mse_comparison['improvement'] = mse_comparison['global'] - mse_comparison['variant']
            mse_comparison['improvement_pct'] = (mse_comparison['improvement'] / mse_comparison['global']) * 100
            mse_comparison = mse_comparison.sort_values('improvement', ascending=False)
            print(mse_comparison)
    
    # Display comprehensive results dataframe
    print(f"\n" + "=" * 80)
    print("COMPREHENSIVE ATE RESULTS - ALL ESTIMATORS")
    print("Complete summary showing all metrics for each estimator and method:")
    
    # Create a clean display dataframe with all relevant columns
    display_df = estimator_summary_df[['estimator', 'method', 'bias', 'variance', 'mse', 
                                      'mean_true_ate', 'mean_estimated_ate', 'n_seeds', 'total_samples']].copy()
    
    # Round numerical values for better display
    numeric_cols = ['bias', 'variance', 'mse', 'mean_true_ate', 'mean_estimated_ate']
    if not display_df.empty:
        display_df[numeric_cols] = display_df[numeric_cols].round(6)
    
    # Sort by estimator and method for organized display
    display_df = display_df.sort_values(['estimator', 'method'])
    
    print(f"\nTotal estimators: {len(all_estimators)}")
    print(f"Methods compared: Global vs Variant")
    print(f"Total rows: {len(display_df)}")
    print()
    
    print(display_df)
    
    return estimator_summary_df, all_ate_results


def calculate_ite_mse_pehe(df_global, df_variant):
    """
    Calculate ITE MSE and PEHE for global and variant-specific methods, and perform
    a bias-variance decomposition for ITE (per-instance across seeds) for each estimator.

    The decomposition follows the user's instructions: for each instance (aligned by sample
    position across seeds), compute the average prediction across seeds, the bias for that
    instance (avg_pred - true), and the variance across seed predictions. Then average the
    squared biases and the variances across instances to get overall bias^2 and variance.

    Returns:
    - ite_summary_df: DataFrame summarizing ITE MSE and PEHE by estimator and method
    - ite_decomposition_df: DataFrame with bias^2, variance, empirical MSE and theoretical MSE (bias^2+var) per estimator/method
    """
    print("Calculating ITE MSE, PEHE and ITE MSE decomposition (per-instance across seeds)...")
    print("=" * 70)

    rows = []
    decomposition_rows = []

    for method, df in [('global', df_global), ('variant', df_variant)]:
        estimators = sorted(df['estimator'].unique()) if 'estimator' in df.columns else ['__default__']
        for estimator in estimators:
            # Select data for this estimator (or all data if no estimator column)
            if 'estimator' in df.columns:
                sub_df = df[df['estimator'] == estimator].copy()
            else:
                sub_df = df.copy()

            if len(sub_df) == 0:
                continue

            # Basic population-level ITE MSE and PEHE (as before)
            ite_errors = sub_df['ite_pred'].values - sub_df['ite_real'].values
            mse = np.mean(ite_errors ** 2)
            pehe = np.sqrt(mse)

            rows.append({
                'method': method,
                'estimator': estimator,
                'ite_mse': mse,
                'pehe': pehe,
                'n_samples': len(sub_df)
            })

            # Now perform the per-instance (across seeds) bias-variance decomposition
            # Requirement: the order of samples in each seed is the same. We therefore align
            # instances by their position within each seed. Create a position index per seed.
            if 'seed' not in sub_df.columns:
                # Cannot perform decomposition without seed information
                decomposition_rows.append({
                    'method': method,
                    'estimator': estimator,
                    'bias_squared': np.nan,
                    'variance': np.nan,
                    'mse_empirical': mse,
                    'mse_theoretical': np.nan,
                    'n_instances': np.nan,
                    'n_seeds': 0
                })
                continue

            tmp = sub_df.copy()
            # Assign position within each seed (0..n-1) based on existing order in the dataframe
            tmp['pos'] = tmp.groupby('seed').cumcount()

            # Pivot to get predictions per seed in columns; index is pos (instance position)
            preds = tmp.pivot(index='pos', columns='seed', values='ite_pred')

            # Get true ITE per pos (should be identical across seeds). Use first non-null per pos.
            true_vals = tmp.drop_duplicates(subset=['seed', 'pos']).groupby('pos')['ite_real'].first()

            # Align preds and true_vals by pos index intersection
            common_index = preds.index.intersection(true_vals.index)
            preds = preds.loc[common_index]
            true_vals = true_vals.loc[common_index]

            if len(common_index) == 0:
                decomposition_rows.append({
                    'method': method,
                    'estimator': estimator,
                    'bias_squared': np.nan,
                    'variance': np.nan,
                    'mse_empirical': mse,
                    'mse_theoretical': np.nan,
                    'n_instances': 0,
                    'n_seeds': preds.shape[1]
                })
                continue

            # Average prediction across seeds for each instance (pos)
            avg_pred = preds.mean(axis=1)

            # Bias per instance = avg_pred - true
            bias_inst = avg_pred - true_vals

            # Variance per instance = variance of seed predictions around avg_pred
            # use ddof=0 to match population variance as requested (can use .var(axis=1) which defaults to ddof=1 for pandas, so use numpy)
            var_inst = preds.apply(lambda row: np.nanvar(row.values, ddof=0), axis=1)

            # Overall (average across instances)
            bias_squared_overall = np.nanmean(bias_inst.values ** 2)
            variance_overall = np.nanmean(var_inst.values)

            # Empirical MSE per instance (average over seeds of squared error), then mean across instances
            # Equivalent: mse_empirical = mean_over_instances( mean_over_seeds( (pred - true)^2 ) )
            sq_err_per_seed = (preds.sub(true_vals, axis=0) ** 2)
            mse_per_instance = sq_err_per_seed.mean(axis=1)
            mse_empirical = np.nanmean(mse_per_instance.values)

            mse_theoretical = bias_squared_overall + variance_overall

            decomposition_rows.append({
                'method': method,
                'estimator': estimator,
                'bias_squared': bias_squared_overall,
                'variance': variance_overall,
                'mse_empirical': mse_empirical,
                'mse_theoretical': mse_theoretical,
                'mse_difference': abs(mse_empirical - mse_theoretical),
                'n_instances': int(len(common_index)),
                'n_seeds': int(preds.shape[1])
            })

    ite_summary_df = pd.DataFrame(rows)
    ite_decomposition_df = pd.DataFrame(decomposition_rows)

    print("\nITE Summary (MSE and PEHE):")
    print(ite_summary_df[['method', 'estimator', 'ite_mse', 'pehe', 'n_samples']])

    print("\nITE MSE Decomposition (per-instance across seeds):")
    print(ite_decomposition_df[['method', 'estimator', 'bias_squared', 'variance', 'mse_empirical', 'mse_theoretical', 'mse_difference', 'n_instances', 'n_seeds']])

    return ite_summary_df, ite_decomposition_df


def calculate_ite_mse_pehe_no_estimator(df_global, df_variant, best_estimator_name):
    """
    Same as calculate_ite_mse_pehe but IGNORE any 'estimator' column:
    treat all rows as coming from a single estimator/method. Returns the same two
    DataFrames: population-level ITE summary and per-instance decomposition.
    """
    print("Calculating ITE MSE, PEHE and per-instance decomposition (NO estimator column)...")
    print("=" * 70)

    rows = []
    decomposition_rows = []

    for method, df in [('global', df_global), ('variant', df_variant)]:
        # Ignore any 'estimator' column: operate on the full dataframe for the method
        sub_df = df.copy()
        if len(sub_df) == 0:
            continue

        # Population-level ITE MSE and PEHE
        ite_errors = sub_df['ite_pred'].values - sub_df['ite_real'].values
        mse = np.mean(ite_errors ** 2)
        pehe = np.sqrt(mse)

        rows.append({
            'method': method,
            'estimator': best_estimator_name,
            'ite_mse': mse,
            'pehe': pehe,
            'n_samples': len(sub_df)
        })

        # Per-instance decomposition across seeds. Requires 'seed' column as before.
        if 'seed' not in sub_df.columns:
            decomposition_rows.append({
                'method': method,
                'estimator': best_estimator_name,
                'bias_squared': np.nan,
                'variance': np.nan,
                'mse_empirical': mse,
                'mse_theoretical': np.nan,
                'mse_difference': np.nan,
                'n_instances': np.nan,
                'n_seeds': 0,
            })
            continue

        tmp = sub_df.copy()
        tmp['pos'] = tmp.groupby('seed').cumcount()
        preds = tmp.pivot(index='pos', columns='seed', values='ite_pred')
        true_vals = tmp.drop_duplicates(subset=['seed', 'pos']).groupby('pos')['ite_real'].first()

        common_index = preds.index.intersection(true_vals.index)
        preds = preds.loc[common_index]
        true_vals = true_vals.loc[common_index]

        if len(common_index) == 0:
            decomposition_rows.append({
                'method': method,
                'estimator': best_estimator_name,
                'bias_squared': np.nan,
                'variance': np.nan,
                'mse_empirical': mse,
                'mse_theoretical': np.nan,
                'mse_difference': np.nan,
                'n_instances': 0,
                'n_seeds': preds.shape[1] if preds.shape[1] is not None else 0,
            })
            continue

        avg_pred = preds.mean(axis=1)
        bias_inst = avg_pred - true_vals
        var_inst = preds.apply(lambda row: np.nanvar(row.values, ddof=0), axis=1)

        bias_squared_overall = np.nanmean(bias_inst.values ** 2)
        variance_overall = np.nanmean(var_inst.values)

        sq_err_per_seed = (preds.sub(true_vals, axis=0) ** 2)
        mse_per_instance = sq_err_per_seed.mean(axis=1)
        mse_empirical = np.nanmean(mse_per_instance.values)

        mse_theoretical = bias_squared_overall + variance_overall

        decomposition_rows.append({
            'method': method,
            'estimator': best_estimator_name,
            'bias_squared': bias_squared_overall,
            'variance': variance_overall,
            'mse_empirical': mse_empirical,
            'mse_theoretical': mse_theoretical,
            'mse_difference': abs(mse_empirical - mse_theoretical),
            'n_instances': int(len(common_index)),
            'n_seeds': int(preds.shape[1])
        })

    ite_summary_df = pd.DataFrame(rows)
    ite_decomposition_df = pd.DataFrame(decomposition_rows)

    print("\nITE Summary (NO estimator):")
    print(ite_summary_df[['method', 'estimator', 'ite_mse', 'pehe', 'n_samples']])

    print("\nITE Decomposition (NO estimator) - bias^2, variance, empirical MSE, theoretical MSE:")
    print(ite_decomposition_df[['method', 'estimator', 'bias_squared', 'variance', 'mse_empirical', 'mse_theoretical', 'mse_difference', 'n_instances', 'n_seeds']])

    return ite_summary_df, ite_decomposition_df


def perform_ate_mse_statistical_tests(ate_by_seed_df, ate_summary_df, 
                                     df_best_global=None, df_best_variant=None, 
                                     best_estimator_name="Best Estimators Selection Per Seed",
                                     alpha=0.05):
    """
    Perform one-sided t-tests comparing ATE MSE between variant and global methods.

    Parameters:
    - ate_by_seed_df: DataFrame with columns ['estimator', 'seed', 'method', 'ate_squared_error']
    - ate_summary_df: DataFrame with summary statistics
    - alpha: significance level (default: 0.05)

    Returns:
    - statistical_results_df: DataFrame with test results and formatted table
    """
    print("Performing Statistical Significance Tests for ATE MSE Comparison")
    print("=" * 70)
    print(f"Null Hypothesis (H₀): MSE_variant ≥ MSE_global")
    print(f"Alternative Hypothesis (H₁): MSE_variant < MSE_global")
    print(f"Significance level: α = {alpha}")
    print()

    required_cols = {"estimator", "seed", "method", "ate_squared_error"}
    missing = required_cols - set(ate_by_seed_df.columns)
    if missing:
        raise ValueError(f"ate_by_seed_df is missing required columns: {missing}")

    # Start from the input per-seed MSE table
    df = ate_by_seed_df.copy()

    # Optionally add BEST_ESTIMATOR_NAME from DF_BEST_GLOBAL / DF_BEST_VARIANT
    if best_estimator_name not in df["estimator"].unique():
        if df_best_global is not None and df_best_variant is not None:
            def _build_best_df(source_df, method_label):
                if source_df is None or len(source_df) == 0:
                    return pd.DataFrame(
                        columns=["estimator", "seed", "method", "ate_squared_error"]
                    )
                # ATE per seed
                grouped = (
                    source_df.groupby("seed", as_index=False)[["ite_real", "ite_pred"]]
                    .mean()
                )
                ate_error = grouped["ite_pred"] - grouped["ite_real"]
                return pd.DataFrame(
                    {
                        "estimator": best_estimator_name,
                        "seed": grouped["seed"],
                        "method": method_label,
                        "ate_squared_error": ate_error ** 2,
                    }
                )

            best_global_df = _build_best_df(df_best_global, "global")
            best_variant_df = _build_best_df(df_best_variant, "variant")

            if not best_global_df.empty or not best_variant_df.empty:
                df = pd.concat([df, best_global_df, best_variant_df], ignore_index=True)

    # Aggregate per (estimator, method, seed) – harmless if already at that granularity
    df_grouped = (
        df.groupby(["estimator", "method", "seed"], as_index=False)["ate_squared_error"]
        .mean()
    )

    # Pivot so we have paired global/variant MSE per seed for each estimator
    pivot = (
        df_grouped
        .pivot_table(
            index=["estimator", "seed"],
            columns="method",
            values="ate_squared_error"
        )
        .dropna(subset=["global", "variant"], how="any")
    ).reset_index()

    if pivot.empty:
        print("No paired global/variant data available for any estimator.")
        return pd.DataFrame()

    results = []
    estimators = sorted(pivot["estimator"].unique())

    for estimator in estimators:
        sub = pivot[pivot["estimator"] == estimator]

        global_data = sub["global"].to_numpy()
        variant_data = sub["variant"].to_numpy()

        if len(global_data) == 0 or len(variant_data) == 0:
            print(f"  Warning: No data found for {estimator}")
            continue

        # They are already paired per seed
        print(f"\nAnalyzing {estimator}...")
        print(f"  Sample size: {len(global_data)} seeds each")
        print(f"  Global MSE: {global_data.mean():.4f} (std: {global_data.std(ddof=1):.4f})")
        print(f"  Variant MSE: {variant_data.mean():.4f} (std: {variant_data.std(ddof=1):.4f})")

        # One-sided paired t-test via differences (variant - global)
        differences = variant_data - global_data
        mean_diff = differences.mean()

        # Two-sided t-test for mean difference = 0
        t_stat, p_value_two_sided = stats.ttest_1samp(differences, 0.0)

        # Convert to one-sided p-value for H1: mean_diff < 0
        if t_stat < 0:
            p_value = p_value_two_sided / 2.0
        else:
            p_value = 1.0 - (p_value_two_sided / 2.0)

        # Effect size (Cohen's d) – use pooled SD of the two groups
        pooled_std = np.sqrt((global_data.std(ddof=1) ** 2 + variant_data.std(ddof=1) ** 2) / 2.0)
        effect_size = abs(mean_diff) / pooled_std if pooled_std > 0 else 0.0

        # Calculate 95% Confidence Intervals for MSE values
        n_seeds = len(global_data)
        confidence_level = 0.95
        alpha_ci = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha_ci/2, df=n_seeds-1)
        
        # Global MSE CI
        global_mean_mse = global_data.mean()
        global_se = global_data.std(ddof=1) / np.sqrt(n_seeds)
        global_ci_lower = global_mean_mse - t_critical * global_se
        global_ci_upper = global_mean_mse + t_critical * global_se
        global_ci = f"[{global_ci_lower:.1f}, {global_ci_upper:.1f}]"
        
        # Variant MSE CI
        variant_mean_mse = variant_data.mean()
        variant_se = variant_data.std(ddof=1) / np.sqrt(n_seeds)
        variant_ci_lower = variant_mean_mse - t_critical * variant_se
        variant_ci_upper = variant_mean_mse + t_critical * variant_se
        variant_ci = f"[{variant_ci_lower:.1f}, {variant_ci_upper:.1f}]"

        # Default summary from stats DF if available
        summary_global = ate_summary_df[
            (ate_summary_df["estimator"] == estimator)
            & (ate_summary_df["method"] == "global")
        ]
        summary_variant = ate_summary_df[
            (ate_summary_df["estimator"] == estimator)
            & (ate_summary_df["method"] == "variant")
        ]

        if not summary_global.empty and not summary_variant.empty:
            global_mse = float(summary_global["mse"].iloc[0])
            variant_mse = float(summary_variant["mse"].iloc[0])
            global_bias_sq = float(summary_global["bias_squared"].iloc[0])
            variant_bias_sq = float(summary_variant["bias_squared"].iloc[0])
            global_variance = float(summary_global["variance"].iloc[0])
            variant_variance = float(summary_variant["variance"].iloc[0])
        else:
            # Fallback: compute from the per-seed MSEs we have
            global_mse = global_data.mean()
            variant_mse = variant_data.mean()

            # Approximate bias and variance from per-seed errors
            global_ate_errors = np.sqrt(global_data)  # sign lost, same as your original trick
            variant_ate_errors = np.sqrt(variant_data)

            global_bias_sq = (global_ate_errors.mean()) ** 2
            variant_bias_sq = (variant_ate_errors.mean()) ** 2
            global_variance = global_ate_errors.var()
            variant_variance = variant_ate_errors.var()

        # Significance stars
        if p_value < 0.001:
            stars = "***"
        elif p_value < 0.01:
            stars = "**"
        elif p_value < 0.05:
            stars = "*"
        else:
            stars = ""

        results.append(
            {
                "Estimator": estimator,
                "Global_MSE": global_mse,
                "Variant_MSE": variant_mse,
                "Global_MSE_CI": global_ci,
                "Variant_MSE_CI": variant_ci,
                "Improvement": global_mse - variant_mse,
                "Improvement_Pct": ((global_mse - variant_mse) / global_mse * 100.0)
                if global_mse > 0
                else 0.0,
                "Global_Bias²": global_bias_sq,
                "Variant_Bias²": variant_bias_sq,
                "Global_Variance": global_variance,
                "Variant_Variance": variant_variance,
                "Mean_Difference": mean_diff,
                "T_Statistic": t_stat,
                "P_Value": p_value,
                "Effect_Size": effect_size,
                "Significance": stars,
                "N_Seeds": len(differences),
            }
        )

        print(f"  Mean difference (variant - global): {mean_diff:.4f}")
        print(f"  T-statistic: {t_stat:.3f}")
        print(f"  P-value (one-sided): {p_value:.4f} {stars}")
        print(f"  Effect size (Cohen's d): {effect_size:.3f}")
        print(f"  Global MSE 95% CI: {global_ci}")
        print(f"  Variant MSE 95% CI: {variant_ci}")

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No valid results found!")
        return pd.DataFrame()

    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE RESULTS SUMMARY")
    print("=" * 70)

    significant_count = (results_df["P_Value"] < alpha).sum()
    total_count = len(results_df)

    print(f"Total estimators tested: {total_count}")
    print(f"Statistically significant improvements: {significant_count}/{total_count}")
    print(f"Significance level: α = {alpha}")
    print()

    return results_df


def perform_cate_mse_statistical_tests(df_all_global, df_all_variant, 
                                      df_best_global=None, df_best_variant=None,
                                      best_estimator_name="Best Estimators Selection Per Seed",
                                      alpha=0.05):
    """
    Perform one-sided t-tests comparing CATE (ITE) MSE between variant and global methods.

    Parameters:
    - alpha: significance level (default: 0.05)

    Returns:
    - statistical_cate_results_df: DataFrame with test results for CATE analysis
    """
    print("Performing Statistical Significance Tests for CATE (ITE) MSE Comparison")
    print("=" * 75)
    print(f"Null Hypothesis (H₀): MSE_variant ≥ MSE_global")
    print(f"Alternative Hypothesis (H₁): MSE_variant < MSE_global")
    print(f"Significance level: α = {alpha}")
    print("NOTE: Using individual-level ITE MSE (not aggregated ATE)")
    print()

    # ---------- Build base DF with global + variant, all estimators ----------
    def _prep_all_df(df, method_label):
        if df is None or len(df) == 0:
            return pd.DataFrame(
                columns=["estimator", "seed", "method", "ite_error", "ite_sq_error"]
            )
        tmp = df.copy()
        tmp["method"] = method_label
        tmp["ite_error"] = tmp["ite_pred"] - tmp["ite_real"]
        tmp["ite_sq_error"] = tmp["ite_error"] ** 2
        return tmp[["estimator", "seed", "method", "ite_error", "ite_sq_error"]]

    # main tables
    df_global_all = _prep_all_df(df_all_global, "global")
    df_variant_all = _prep_all_df(df_all_variant, "variant")

    all_df = pd.concat([df_global_all, df_variant_all], ignore_index=True)

    # ---------- Add BEST estimator from DF_BEST_* if needed ----------
    if best_estimator_name not in all_df["estimator"].unique():
        def _prep_best_df(df, method_label):
            if df is None or len(df) == 0:
                return pd.DataFrame(
                    columns=["estimator", "seed", "method", "ite_error", "ite_sq_error"]
                )
            tmp = df.copy()
            tmp["estimator"] = best_estimator_name
            tmp["method"] = method_label
            tmp["ite_error"] = tmp["ite_pred"] - tmp["ite_real"]
            tmp["ite_sq_error"] = tmp["ite_error"] ** 2
            return tmp[["estimator", "seed", "method", "ite_error", "ite_sq_error"]]

        best_global_df = _prep_best_df(df_best_global, "global")
        best_variant_df = _prep_best_df(df_best_variant, "variant")

        all_df = pd.concat([all_df, best_global_df, best_variant_df], ignore_index=True)

    # Only keep estimators that appear in both methods
    est_global = set(all_df[all_df["method"] == "global"]["estimator"].unique())
    est_variant = set(all_df[all_df["method"] == "variant"]["estimator"].unique())
    estimators = sorted(est_global & est_variant)

    if not estimators:
        print("No estimators with both global and variant data.")
        return pd.DataFrame()

    # ---------- Per-seed CATE MSE (mean ITE^2 per estimator/method/seed) ----------
    seed_mse = (
        all_df
        .groupby(["estimator", "method", "seed"], as_index=False)["ite_sq_error"]
        .mean()
        .rename(columns={"ite_sq_error": "cate_mse"})
    )

    # Pivot so each row = (estimator, seed) with global + variant CATE MSE
    pivot = (
        seed_mse
        .pivot_table(
            index=["estimator", "seed"],
            columns="method",
            values="cate_mse"
        )
        .dropna(subset=["global", "variant"], how="any")
        .reset_index()
    )

    if pivot.empty:
        print("No paired global/variant CATE MSE data.")
        return pd.DataFrame()

    # ---------- For bias/variance, keep all individual ITE errors ----------
    # (no loops over seeds later – just filter by estimator & method)
    # all_df already has "ite_error"
    results = []

    for est in estimators:
        sub = pivot[pivot["estimator"] == est]

        global_data = sub["global"].to_numpy()
        variant_data = sub["variant"].to_numpy()

        if len(global_data) == 0 or len(variant_data) == 0:
            print(f"  Warning: no paired data for estimator {est}")
            continue

        # Paired per seed
        print(f"\nAnalyzing {est} for CATE (ITE) performance...")
        print(f"  Sample size: {len(global_data)} seeds each")
        print(f"  Global CATE MSE: {global_data.mean():.4f} (std: {global_data.std(ddof=1):.4f})")
        print(f"  Variant CATE MSE: {variant_data.mean():.4f} (std: {variant_data.std(ddof=1):.4f})")

        # differences: variant - global
        differences = variant_data - global_data
        mean_diff = differences.mean()

        # One-sample t-test on differences vs 0
        t_stat, p_two = stats.ttest_1samp(differences, 0.0)

        # One-sided p-value for H1: variant < global
        if t_stat < 0:
            p_value = p_two / 2.0
        else:
            p_value = 1.0 - (p_two / 2.0)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((global_data.std(ddof=1) ** 2 + variant_data.std(ddof=1) ** 2) / 2.0)
        effect_size = abs(mean_diff) / pooled_std if pooled_std > 0 else 0.0

        # 95% CI for MSE per method
        n_seeds = len(global_data)
        confidence_level = 0.95
        alpha_ci = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha_ci / 2.0, df=n_seeds - 1)

        global_mean_mse = global_data.mean()
        global_se = global_data.std(ddof=1) / np.sqrt(n_seeds)
        global_ci_lower = global_mean_mse - t_critical * global_se
        global_ci_upper = global_mean_mse + t_critical * global_se
        global_ci = f"[{global_ci_lower:.1f}, {global_ci_upper:.1f}]"

        variant_mean_mse = variant_data.mean()
        variant_se = variant_data.std(ddof=1) / np.sqrt(n_seeds)
        variant_ci_lower = variant_mean_mse - t_critical * variant_se
        variant_ci_upper = variant_mean_mse + t_critical * variant_se
        variant_ci = f"[{variant_ci_lower:.1f}, {variant_ci_upper:.1f}]"

        # ---------- Bias–variance decomposition using all ITE errors ----------
        est_global_errors = all_df[(all_df["estimator"] == est) & (all_df["method"] == "global")]["ite_error"].to_numpy()
        est_variant_errors = all_df[(all_df["estimator"] == est) & (all_df["method"] == "variant")]["ite_error"].to_numpy()

        global_bias_sq = (est_global_errors.mean()) ** 2 if len(est_global_errors) > 0 else np.nan
        variant_bias_sq = (est_variant_errors.mean()) ** 2 if len(est_variant_errors) > 0 else np.nan
        global_variance = est_global_errors.var() if len(est_global_errors) > 0 else np.nan
        variant_variance = est_variant_errors.var() if len(est_variant_errors) > 0 else np.nan

        # Significance stars
        if p_value < 0.001:
            stars = "***"
        elif p_value < 0.01:
            stars = "**"
        elif p_value < 0.05:
            stars = "*"
        else:
            stars = ""

        results.append(
            {
                "Estimator": est,
                "Global_CATE_MSE": global_mean_mse,
                "Variant_CATE_MSE": variant_mean_mse,
                "Global_CATE_MSE_CI": global_ci,
                "Variant_CATE_MSE_CI": variant_ci,
                "Improvement": global_mean_mse - variant_mean_mse,
                "Improvement_Pct": ((global_mean_mse - variant_mean_mse) / global_mean_mse * 100.0)
                if global_mean_mse > 0
                else 0.0,
                "Global_Bias²": global_bias_sq,
                "Variant_Bias²": variant_bias_sq,
                "Global_Variance": global_variance,
                "Variant_Variance": variant_variance,
                "Mean_Difference": mean_diff,
                "T_Statistic": t_stat,
                "P_Value": p_value,
                "Effect_Size": effect_size,
                "Significance": stars,
                "N_Seeds": len(differences),
            }
        )

        print(f"  Mean difference (variant - global): {mean_diff:.4f}")
        print(f"  T-statistic: {t_stat:.3f}")
        print(f"  P-value (one-sided): {p_value:.4f} {stars}")
        print(f"  Effect size (Cohen's d): {effect_size:.3f}")
        print(f"  Global CATE MSE 95% CI: {global_ci}")
        print(f"  Variant CATE MSE 95% CI: {variant_ci}")

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No valid CATE results found!")
        return pd.DataFrame()

    print("\n" + "=" * 75)
    print("CATE (ITE) STATISTICAL SIGNIFICANCE RESULTS SUMMARY")
    print("=" * 75)

    significant_count = (results_df["P_Value"] < alpha).sum()
    total_count = len(results_df)

    print(f"Total estimators tested: {total_count}")
    print(f"Statistically significant improvements: {significant_count}/{total_count}")
    print(f"Significance level: α = {alpha}")
    print()

    return results_df


def create_comprehensive_statistical_table(statistical_results_df):
    """
    Create the final comprehensive statistical table with exact columns as requested.
    
    Columns:
    - Estimator
    - Global MSE
    - Variant MSE  
    - MSE Improvement (%)
    - Bias² (Global / Variant)
    - Variance (Global / Variant)
    - P-value (one-sided)
    - Effect Size (Cohen's d)
    - 95% CI (Global / Variant)
    
    Parameters:
    - statistical_results_df: DataFrame from perform_ate_mse_statistical_tests function
    
    Returns:
    - final_table: DataFrame with the comprehensive results
    """
    if statistical_results_df.empty:
        print("No statistical results available")
        return pd.DataFrame()
    
    table_data = []
    
    for _, row in statistical_results_df.iterrows():
        # Format MSE values (rounded to whole numbers for readability)
        global_mse = row['Global_MSE']
        variant_mse = row['Variant_MSE']
        
        # Format improvement percentage
        improvement_pct = row['Improvement_Pct']
        
        # Format Bias² values (Global / Variant)
        global_bias_sq = row['Global_Bias²']
        variant_bias_sq = row['Variant_Bias²']
        bias_ratio = f"{global_bias_sq:.0f} / {variant_bias_sq:.0f}"
        
        # Format Variance values (Global / Variant)
        global_variance = row['Global_Variance']
        variant_variance = row['Variant_Variance']
        variance_ratio = f"{global_variance:.0f} / {variant_variance:.0f}"
        
        # Format p-value with significance stars
        p_value = row['P_Value']
        significance = row['Significance']
        p_value_formatted = f"{p_value:.4f}{significance}"
        
        # Create the table row
        table_row = {
            'Estimator': row['Estimator'],
            'Global MSE': f"{global_mse:.0f}",
            'Variant MSE': f"{variant_mse:.0f}",
            'MSE Improvement (%)': f"{improvement_pct:.1f}%",
            'Bias² (Global / Variant)': bias_ratio,
            'Variance (Global / Variant)': variance_ratio,
            'P-value (one-sided)': p_value_formatted,
            'Effect Size (Cohen\'s d)': f"{row['Effect_Size']:.3f}",
            '95% CI (Global / Variant)': f"{row.get('Global_MSE_CI', 'N/A')} / {row.get('Variant_MSE_CI', 'N/A')}"
        }
        
        table_data.append(table_row)
    
    # Create the final table
    final_table = pd.DataFrame(table_data)
    
    # Set estimator as index for better presentation
    final_table_display = final_table.set_index('Estimator')
    
    print("COMPREHENSIVE STATISTICAL RESULTS TABLE")
    print("=" * 100)
    print("Statistical Test: One-sided t-test (H₁: MSE_variant < MSE_global)")
    print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05")
    print("Effect Size interpretation: Small ≥0.2, Medium ≥0.5, Large ≥0.8")
    print("95% CI: Confidence interval for MSE values")
    print()
    
    return final_table_display


def create_comprehensive_cate_statistical_table(statistical_cate_results_df):
    """
    Create the final comprehensive CATE statistical table with exact columns as requested.
    
    Columns:
    - Estimator
    - Global CATE MSE
    - Variant CATE MSE  
    - MSE Improvement (%)
    - Bias² (Global / Variant)
    - Variance (Global / Variant)
    - P-value (one-sided)
    - Effect Size (Cohen's d)
    - 95% CI (Global / Variant)
    
    Parameters:
    - statistical_cate_results_df: DataFrame from perform_cate_mse_statistical_tests function
    
    Returns:
    - final_cate_table: DataFrame with the comprehensive CATE results
    """
    if statistical_cate_results_df.empty:
        print("No CATE statistical results available")
        return pd.DataFrame()
    
    table_data = []
    
    for _, row in statistical_cate_results_df.iterrows():
        # Format MSE values (rounded to whole numbers for readability)
        global_cate_mse = row['Global_CATE_MSE']
        variant_cate_mse = row['Variant_CATE_MSE']
        
        # Format improvement percentage
        improvement_pct = row['Improvement_Pct']
        
        # Format Bias² values (Global / Variant)
        global_bias_sq = row['Global_Bias²']
        variant_bias_sq = row['Variant_Bias²']
        bias_ratio = f"{global_bias_sq:.0f} / {variant_bias_sq:.0f}"
        
        # Format Variance values (Global / Variant)
        global_variance = row['Global_Variance']
        variant_variance = row['Variant_Variance']
        variance_ratio = f"{global_variance:.0f} / {variant_variance:.0f}"
        
        # Format p-value with significance stars
        p_value = row['P_Value']
        significance = row['Significance']
        p_value_formatted = f"{p_value:.4f}{significance}"
        
        # Get confidence intervals
        global_ci = row.get('Global_CATE_MSE_CI', 'N/A')
        variant_ci = row.get('Variant_CATE_MSE_CI', 'N/A')
        
        # Create the table row
        table_row = {
            'Estimator': row['Estimator'],
            'Global CATE MSE': f"{global_cate_mse:.0f}",
            'Variant CATE MSE': f"{variant_cate_mse:.0f}",
            'MSE Improvement (%)': f"{improvement_pct:.1f}%",
            'Bias² (Global / Variant)': bias_ratio,
            'Variance (Global / Variant)': variance_ratio,
            'P-value (one-sided)': p_value_formatted,
            'Effect Size (Cohen\'s d)': f"{row['Effect_Size']:.3f}",
            '95% CI (Global / Variant)': f"{global_ci} / {variant_ci}"
        }
        
        table_data.append(table_row)
    
    # Create the final table
    final_cate_table = pd.DataFrame(table_data)
    
    # Set estimator as index for better presentation
    final_cate_table_display = final_cate_table.set_index('Estimator')
    
    print("COMPREHENSIVE CATE (ITE) STATISTICAL RESULTS TABLE")
    print("=" * 105)
    print("Statistical Test: One-sided t-test (H₁: CATE_MSE_variant < CATE_MSE_global)")
    print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05")
    print("Effect Size interpretation: Small ≥0.2, Medium ≥0.5, Large ≥0.8")
    print("95% CI: Confidence interval for CATE MSE values")
    print("CATE MSE: Mean Squared Error of Individual Treatment Effect predictions")
    print()
    
    return final_cate_table_display


def prepare_dataframes_for_analysis(results_by_seed, seeds):
    """
    Extract and prepare the four main dataframes from multi-seed experiment results.
    
    Parameters:
    - results_by_seed: Dictionary containing results for each seed
    - seeds: List of seeds to process
    
    Returns:
    - Dictionary containing the prepared dataframes with error columns added
    """
    print("Extracting dataframes from results...")
    print("=" * 60)
    
    # Initialize lists to collect dataframes across seeds
    df_best_global_list = []
    df_best_variant_list = []
    df_all_global_list = []
    df_all_variant_list = []
    
    for seed in seeds:
        seed_data = results_by_seed[seed]
        
        # Extract and tag with seed
        for key in ['global_method_best', 'variant_method_best', 'global_method', 'variant_method']:
            if key in seed_data:
                df = seed_data[key].copy()
                df['seed'] = seed
                
                if key == 'global_method_best':
                    df_best_global_list.append(df)
                elif key == 'variant_method_best':
                    df_best_variant_list.append(df)
                elif key == 'global_method':
                    df_all_global_list.append(df)
                elif key == 'variant_method':
                    df_all_variant_list.append(df)
    
    # Combine across seeds
    DF_BEST_GLOBAL = pd.concat(df_best_global_list, ignore_index=True)
    DF_BEST_VARIANT = pd.concat(df_best_variant_list, ignore_index=True)
    DF_ALL_GLOBAL = pd.concat(df_all_global_list, ignore_index=True)
    DF_ALL_VARIANT = pd.concat(df_all_variant_list, ignore_index=True)
    
    print(f"✓ DF_BEST_GLOBAL: {DF_BEST_GLOBAL.shape}")
    print(f"✓ DF_BEST_VARIANT: {DF_BEST_VARIANT.shape}")
    print(f"✓ DF_ALL_GLOBAL: {DF_ALL_GLOBAL.shape}")
    print(f"✓ DF_ALL_VARIANT: {DF_ALL_VARIANT.shape}")
    
    # Display basic statistics
    print(f"\nSeeds in data: {sorted(DF_BEST_GLOBAL['seed'].unique())}")
    print(f"Variants: {sorted(DF_BEST_GLOBAL['variant'].unique())}")
    if 'estimator' in DF_ALL_GLOBAL.columns:
        print(f"Estimators: {sorted(DF_ALL_GLOBAL['estimator'].unique())}")
    
    # Check required columns
    required_cols = ['seed', 'variant', 'ite_real', 'ite_pred']
    missing_cols = []
    for df_name, df in [('DF_BEST_GLOBAL', DF_BEST_GLOBAL), ('DF_BEST_VARIANT', DF_BEST_VARIANT), 
                        ('DF_ALL_GLOBAL', DF_ALL_GLOBAL), ('DF_ALL_VARIANT', DF_ALL_VARIANT)]:
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            missing_cols.append(f"{df_name}: missing {missing}")
    
    if missing_cols:
        print(f"\n⚠️  Missing columns:")
        for missing in missing_cols:
            print(f"   {missing}")
    else:
        print(f"\n✓ All required columns present in all dataframes")
    
    # Add error calculation columns and identify fallbacks
    DF_BEST_GLOBAL = add_error_columns(DF_BEST_GLOBAL)
    DF_BEST_VARIANT = add_error_columns(DF_BEST_VARIANT)
    DF_ALL_GLOBAL = add_error_columns(DF_ALL_GLOBAL)
    DF_ALL_VARIANT = add_error_columns(identify_fallback_variants(DF_ALL_VARIANT))
    
    print(f"✓ Added error columns: ite_error, ite_squared_error, ite_abs_error")
    
    # Identify true variant-specific models (exclude fallbacks)
    DF_ALL_VARIANT_TRUE = DF_ALL_VARIANT[~DF_ALL_VARIANT['is_fallback']].copy()
    print(f"✓ True variant-specific models: {DF_ALL_VARIANT_TRUE.shape} (excluding {DF_ALL_VARIANT['is_fallback'].sum()} fallback cases)")
    
    return {
        'DF_BEST_GLOBAL': DF_BEST_GLOBAL,
        'DF_BEST_VARIANT': DF_BEST_VARIANT,
        'DF_ALL_GLOBAL': DF_ALL_GLOBAL,
        'DF_ALL_VARIANT': DF_ALL_VARIANT,
        'DF_ALL_VARIANT_TRUE': DF_ALL_VARIANT_TRUE
    }
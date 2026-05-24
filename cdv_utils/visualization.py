"""
Visualization Module for Causal Analysis

This module provides plotting and visualization functions for analyzing
causal inference results, including bias-variance trade-off plots and
other statistical visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def plot_ate_bias_variance_tradeoff(ate_summary_df, best_estimator_name="Best Estimators Selection Per Seed", 
                                   title_prefix="ATE", save_path=None):
    """
    Create a single bias-variance scatter plot comparing global vs variant methods for ATE.
    Uses the ate summary dataframe from calculate_ate_by_estimator function.
    Values normalized to [0, 1] scale with 10% padding.
    
    Parameters:
    - ate_summary_df: Summary dataframe containing bias, variance, mse columns by estimator and method
    - best_estimator_name: Name of the best estimator to highlight
    - title_prefix: Prefix for the plot title (default: "ATE")
    - save_path: Optional path to save the plot
    """
    
    # Get global and variant data separately
    global_data = ate_summary_df[ate_summary_df['method'] == 'global'].copy()
    variant_data = ate_summary_df[ate_summary_df['method'] == 'variant'].copy()
    
    if len(global_data) == 0 or len(variant_data) == 0:
        print("Warning: Missing data for global or variant methods")
        return
    
    # Combine for normalization
    combined = pd.concat([global_data, variant_data])
    bias_min, bias_max = combined['bias_squared'].min(), combined['bias_squared'].max()
    var_min, var_max = combined['variance'].min(), combined['variance'].max()
    
    # Add padding
    padding = 0.1
    bias_range = bias_max - bias_min if bias_max != bias_min else 1
    var_range = var_max - var_min if var_max != var_min else 1
    
    bias_min -= bias_range * padding
    bias_max += bias_range * padding
    var_min -= var_range * padding
    var_max += var_range * padding
    
    def normalize(value, min_val, max_val):
        return 0.5 if max_val == min_val else (value - min_val) / (max_val - min_val)
    
    # Setup plot
    all_estimators = sorted(combined['estimator'].unique())
    markers = ['*', 's', 'o', 'D', 'v', 'p', '^', 'h', '<', '>']
    estimator_markers = {est: markers[i % len(markers)] for i, est in enumerate(all_estimators)}
    method_colors = {'global': '#1f77b4', 'variant': '#ff7f0e'}
    method_labels = {'global': 'Global Method', 'variant': 'CPV Partition Method'}
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot points for each method
    for method, color in method_colors.items():
        method_data = combined[combined['method'] == method]
        for _, row in method_data.iterrows():
            norm_bias = normalize(row['bias_squared'], bias_min, bias_max)
            norm_var = normalize(row['variance'], var_min, var_max)
            est = row['estimator']
            is_best = (est == best_estimator_name)

            # Bold effect: larger marker + thicker edge
            size = 150 if is_best else 100
            lw   = 1 if is_best else 0.5
            ax.scatter(norm_bias, norm_var,
                      s=size, alpha=0.85, color=color,
                      marker=estimator_markers[row['estimator']],
                      edgecolors='black', linewidths=lw)
    
    ax.set_xlabel(f'Bias² ({title_prefix}) - Normalized', fontsize=13)
    ax.set_ylabel(f'Variance ({title_prefix}) - Normalized', fontsize=13)
    ax.set_title(f'{title_prefix} Bias-Variance Tradeoff (Global vs CPV Partition)', 
                 fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Legend
    legend_elements = []
    # Method colors
    for method, color in method_colors.items():
        legend_elements.append(Patch(facecolor=color, edgecolor='black', 
                                   label=method_labels[method]))
    
    # Separator
    legend_elements.append(Patch(facecolor='none', edgecolor='none', label=''))
    
    # Estimator markers
    for est in all_estimators:
        legend_elements.append(Line2D([0], [0], marker=estimator_markers[est],
                                     color='gray', linestyle='None', markersize=10,
                                     label=est, markeredgecolor='black', markeredgewidth=1.0))
    
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), 
              fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_cate_bias_variance_tradeoff(cate_summary_df, best_estimator_name="Best Estimators Selection Per Seed",
                                    title_prefix="CATE", save_path=None):
    """
    Create a single bias-variance scatter plot comparing global vs variant methods for CATE.
    Uses the cate summary dataframe from calculate_cate_by_estimator function.
    Values normalized to [0, 1] scale with 10% padding.
    
    Parameters:
    - cate_summary_df: Summary dataframe containing bias, variance, mse columns by estimator and method
    - best_estimator_name: Name of the best estimator to highlight
    - title_prefix: Prefix for the plot title (default: "CATE")
    - save_path: Optional path to save the plot
    """
    
    # Get global and variant data separately
    global_data = cate_summary_df[cate_summary_df['method'] == 'global'].copy()
    variant_data = cate_summary_df[cate_summary_df['method'] == 'variant'].copy()
    
    if len(global_data) == 0 or len(variant_data) == 0:
        print("Warning: Missing data for global or variant methods")
        return
    
    # Combine for normalization
    combined = pd.concat([global_data, variant_data])
    bias_min, bias_max = combined['bias_squared'].min(), combined['bias_squared'].max()
    var_min, var_max = combined['variance'].min(), combined['variance'].max()
    
    # Add padding
    padding = 0.1
    bias_range = bias_max - bias_min if bias_max != bias_min else 1
    var_range = var_max - var_min if var_max != var_min else 1
    
    bias_min -= bias_range * padding
    bias_max += bias_range * padding
    var_min -= var_range * padding
    var_max += var_range * padding
    
    def normalize(value, min_val, max_val):
        return 0.5 if max_val == min_val else (value - min_val) / (max_val - min_val)
    
    # Setup plot
    all_estimators = sorted(combined['estimator'].unique())
    markers = ['*', 's', 'o', 'D', 'v', 'p', '^', 'h', '<', '>']
    estimator_markers = {est: markers[i % len(markers)] for i, est in enumerate(all_estimators)}
    method_colors = {'global': '#1f77b4', 'variant': '#ff7f0e'}
    method_labels = {'global': 'Global Method', 'variant': 'CPV Partition Method'}
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot points for each method
    for method, color in method_colors.items():
        method_data = combined[combined['method'] == method]
        for _, row in method_data.iterrows():
            norm_bias = normalize(row['bias_squared'], bias_min, bias_max)
            norm_var = normalize(row['variance'], var_min, var_max)
            est = row['estimator']
            is_best = (est == best_estimator_name)

            # Bold effect: larger marker + thicker edge
            size = 150 if is_best else 100
            lw   = 1 if is_best else 0.5
            ax.scatter(norm_bias, norm_var,
                      s=size, alpha=0.85, color=color,
                      marker=estimator_markers[row['estimator']],
                      edgecolors='black', linewidths=lw)
    
    ax.set_xlabel(f'Bias² ({title_prefix}) - Normalized', fontsize=13)
    ax.set_ylabel(f'Variance ({title_prefix}) - Normalized', fontsize=13)
    ax.set_title(f'{title_prefix} Bias-Variance Tradeoff (Global vs CPV Partition)', 
                 fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Legend
    legend_elements = []
    # Method colors
    for method, color in method_colors.items():
        legend_elements.append(Patch(facecolor=color, edgecolor='black', 
                                   label=method_labels[method]))
    
    # Separator
    legend_elements.append(Patch(facecolor='none', edgecolor='none', label=''))
    
    # Estimator markers
    for est in all_estimators:
        legend_elements.append(Line2D([0], [0], marker=estimator_markers[est],
                                     color='gray', linestyle='None', markersize=10,
                                     label=est, markeredgecolor='black', markeredgewidth=1.0))
    
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), 
              fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def setup_plotting_style():
    """Set up the plotting style for consistent visualization"""
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    print("✓ Plotting style configured")


def plot_mse_comparison(comparison_df, method1='global', method2='variant', 
                       metric='mse', title=None, save_path=None):
    """
    Create a comparison plot between two methods for a specific metric.
    
    Parameters:
    - comparison_df: DataFrame with methods as columns and estimators as index
    - method1: First method to compare (default: 'global')
    - method2: Second method to compare (default: 'variant')  
    - metric: Metric to compare (default: 'mse')
    - title: Custom title for the plot
    - save_path: Optional path to save the plot
    """
    if metric not in comparison_df.columns.get_level_values(0):
        print(f"Metric '{metric}' not found in comparison dataframe")
        return
    
    if method1 not in comparison_df[metric].columns or method2 not in comparison_df[metric].columns:
        print(f"Methods '{method1}' or '{method2}' not found in comparison dataframe")
        return
    
    data = comparison_df[metric][[method1, method2]].copy()
    data['improvement'] = data[method1] - data[method2]
    data['improvement_pct'] = (data['improvement'] / data[method1]) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot absolute values
    estimators = data.index
    x_pos = np.arange(len(estimators))
    
    ax1.bar(x_pos - 0.2, data[method1], 0.4, label=method1.title(), alpha=0.8)
    ax1.bar(x_pos + 0.2, data[method2], 0.4, label=method2.title(), alpha=0.8)
    
    ax1.set_xlabel('Estimators')
    ax1.set_ylabel(f'{metric.upper()}')
    ax1.set_title(f'{metric.upper()} Comparison: {method1.title()} vs {method2.title()}')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(estimators, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot improvement percentage
    colors = ['green' if x > 0 else 'red' for x in data['improvement_pct']]
    bars = ax2.bar(x_pos, data['improvement_pct'], color=colors, alpha=0.7)
    
    ax2.set_xlabel('Estimators')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title(f'{metric.upper()} Improvement: {method2.title()} vs {method1.title()}')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(estimators, rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -15),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=8)
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_statistical_results_summary(statistical_results_df, title="Statistical Results Summary", 
                                    save_path=None):
    """
    Create visualization of statistical test results.
    
    Parameters:
    - statistical_results_df: DataFrame with statistical test results
    - title: Title for the plot
    - save_path: Optional path to save the plot
    """
    if statistical_results_df.empty:
        print("No statistical results to plot")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # P-values
    estimators = statistical_results_df['Estimator']
    p_values = statistical_results_df['P_Value']
    colors = ['green' if p < 0.05 else 'orange' if p < 0.1 else 'red' for p in p_values]
    
    bars1 = ax1.bar(range(len(estimators)), p_values, color=colors, alpha=0.7)
    ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.8, label='α = 0.05')
    ax1.axhline(y=0.01, color='darkred', linestyle='--', alpha=0.8, label='α = 0.01')
    ax1.set_xlabel('Estimators')
    ax1.set_ylabel('P-value')
    ax1.set_title('Statistical Significance (One-sided t-test)')
    ax1.set_xticks(range(len(estimators)))
    ax1.set_xticklabels(estimators, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Effect sizes
    effect_sizes = statistical_results_df['Effect_Size']
    colors_effect = ['darkgreen' if es >= 0.8 else 'green' if es >= 0.5 else 'orange' if es >= 0.2 else 'red' 
                    for es in effect_sizes]
    
    bars2 = ax2.bar(range(len(estimators)), effect_sizes, color=colors_effect, alpha=0.7)
    ax2.axhline(y=0.2, color='orange', linestyle='--', alpha=0.8, label='Small (0.2)')
    ax2.axhline(y=0.5, color='green', linestyle='--', alpha=0.8, label='Medium (0.5)')
    ax2.axhline(y=0.8, color='darkgreen', linestyle='--', alpha=0.8, label='Large (0.8)')
    ax2.set_xlabel('Estimators')
    ax2.set_ylabel('Effect Size (Cohen\'s d)')
    ax2.set_title('Effect Size of Improvement')
    ax2.set_xticks(range(len(estimators)))
    ax2.set_xticklabels(estimators, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # MSE Improvement percentages
    improvements = statistical_results_df['Improvement_Pct']
    colors_imp = ['green' if imp > 0 else 'red' for imp in improvements]
    
    bars3 = ax3.bar(range(len(estimators)), improvements, color=colors_imp, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_xlabel('Estimators')
    ax3.set_ylabel('MSE Improvement (%)')
    ax3.set_title('MSE Improvement: Variant vs Global')
    ax3.set_xticks(range(len(estimators)))
    ax3.set_xticklabels(estimators, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # MSE values comparison
    global_mse = statistical_results_df['Global_MSE']
    variant_mse = statistical_results_df['Variant_MSE']
    
    x_pos = np.arange(len(estimators))
    width = 0.35
    
    ax4.bar(x_pos - width/2, global_mse, width, label='Global', alpha=0.8, color='skyblue')
    ax4.bar(x_pos + width/2, variant_mse, width, label='Variant', alpha=0.8, color='lightcoral')
    
    ax4.set_xlabel('Estimators')
    ax4.set_ylabel('MSE')
    ax4.set_title('MSE Comparison: Global vs Variant')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(estimators, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_results_visualization_dashboard(ate_statistical_results, cate_statistical_results, 
                                         ate_summary_df, cate_summary_df, save_path=None):
    """
    Create a comprehensive dashboard showing all key results visualizations.
    
    Parameters:
    - ate_statistical_results: ATE statistical test results DataFrame
    - cate_statistical_results: CATE statistical test results DataFrame  
    - ate_summary_df: ATE bias-variance summary DataFrame
    - cate_summary_df: CATE bias-variance summary DataFrame
    - save_path: Optional path to save the dashboard
    """
    fig = plt.figure(figsize=(20, 16))
    
    # Create a grid layout
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])
    
    # ATE Bias-Variance plot (subplot 1)
    ax1 = fig.add_subplot(gs[0, 0])
    plt.sca(ax1)
    plot_ate_bias_variance_tradeoff(ate_summary_df, title_prefix="ATE")
    
    # CATE Bias-Variance plot (subplot 2) 
    ax2 = fig.add_subplot(gs[0, 1])
    plt.sca(ax2)
    plot_cate_bias_variance_tradeoff(cate_summary_df, title_prefix="CATE")
    
    # ATE Statistical Results (subplot 3)
    ax3 = fig.add_subplot(gs[1, 0])
    if not ate_statistical_results.empty:
        estimators = ate_statistical_results['Estimator']
        improvements = ate_statistical_results['Improvement_Pct']
        p_values = ate_statistical_results['P_Value']
        
        colors = ['green' if p < 0.05 else 'orange' if p < 0.1 else 'red' for p in p_values]
        bars = ax3.bar(range(len(estimators)), improvements, color=colors, alpha=0.7)
        
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_xlabel('Estimators')
        ax3.set_ylabel('ATE MSE Improvement (%)')
        ax3.set_title('ATE MSE Improvement: Variant vs Global')
        ax3.set_xticks(range(len(estimators)))
        ax3.set_xticklabels(estimators, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
    
    # CATE Statistical Results (subplot 4)
    ax4 = fig.add_subplot(gs[1, 1])
    if not cate_statistical_results.empty:
        estimators = cate_statistical_results['Estimator']
        improvements = cate_statistical_results['Improvement_Pct']
        p_values = cate_statistical_results['P_Value']
        
        colors = ['green' if p < 0.05 else 'orange' if p < 0.1 else 'red' for p in p_values]
        bars = ax4.bar(range(len(estimators)), improvements, color=colors, alpha=0.7)
        
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.set_xlabel('Estimators')
        ax4.set_ylabel('CATE MSE Improvement (%)')
        ax4.set_title('CATE MSE Improvement: Variant vs Global')
        ax4.set_xticks(range(len(estimators)))
        ax4.set_xticklabels(estimators, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
    
    # Summary Statistics (subplot 5-6 combined)
    ax5 = fig.add_subplot(gs[2, :])
    
    # Create summary table
    summary_data = []
    
    if not ate_statistical_results.empty:
        ate_significant = (ate_statistical_results['P_Value'] < 0.05).sum()
        ate_total = len(ate_statistical_results)
        ate_avg_improvement = ate_statistical_results['Improvement_Pct'].mean()
        summary_data.append(['ATE', ate_total, ate_significant, f"{ate_avg_improvement:.1f}%"])
    
    if not cate_statistical_results.empty:
        cate_significant = (cate_statistical_results['P_Value'] < 0.05).sum()
        cate_total = len(cate_statistical_results) 
        cate_avg_improvement = cate_statistical_results['Improvement_Pct'].mean()
        summary_data.append(['CATE', cate_total, cate_significant, f"{cate_avg_improvement:.1f}%"])
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data, 
                                columns=['Metric', 'Total Estimators', 'Significant (p<0.05)', 'Avg Improvement'])
        
        ax5.axis('tight')
        ax5.axis('off')
        table = ax5.table(cellText=summary_df.values, colLabels=summary_df.columns,
                         cellLoc='center', loc='center', bbox=[0.2, 0.3, 0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        ax5.set_title('Summary of Statistical Results', fontsize=14, fontweight='bold', pad=20)
    
    fig.suptitle('Causal Analysis Results Dashboard', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


# =============================================================================
# SYNTHETIC EXPERIMENT VISUALIZATIONS
# =============================================================================


def plot_scissors_chart(results_by_alpha, metric='ate', save_path=None, figsize=(10, 6)):
    """
    Plot MSE vs heterogeneity (alpha) for Global vs CDV methods.
    
    Shows the "scissors" effect: as alpha increases, global model degrades
    while CDV method remains stable, producing diverging lines.
    
    Parameters:
    -----------
    results_by_alpha : dict
        Dictionary mapping alpha values to analysis results.
        Each value should have keys: 'global_mse_per_seed', 'cdv_mse_per_seed'
        (lists of per-seed MSE values).
    metric : str
        'ate' or 'cate' - which metric to plot
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    alphas = sorted(results_by_alpha.keys())
    
    global_means = []
    global_cis = []
    cdv_means = []
    cdv_cis = []
    
    for alpha in alphas:
        result = results_by_alpha[alpha]
        
        g_vals = np.array(result['global_mse_per_seed'])
        c_vals = np.array(result['cdv_mse_per_seed'])
        
        global_means.append(np.mean(g_vals))
        cdv_means.append(np.mean(c_vals))
        
        # 95% CI
        n = len(g_vals)
        global_cis.append(1.96 * np.std(g_vals) / np.sqrt(n))
        cdv_cis.append(1.96 * np.std(c_vals) / np.sqrt(n))
    
    alphas = np.array(alphas)
    global_means = np.array(global_means)
    cdv_means = np.array(cdv_means)
    global_cis = np.array(global_cis)
    cdv_cis = np.array(cdv_cis)
    
    # Plot lines with CI bands
    ax.plot(alphas, global_means, 'o-', color='#d62728', linewidth=2.5, markersize=8, label='Global Model')
    ax.fill_between(alphas, global_means - global_cis, global_means + global_cis, 
                    color='#d62728', alpha=0.15)
    
    ax.plot(alphas, cdv_means, 's-', color='#1f77b4', linewidth=2.5, markersize=8, label='CDV Model')
    ax.fill_between(alphas, cdv_means - cdv_cis, cdv_means + cdv_cis,
                    color='#1f77b4', alpha=0.15)
    
    ax.set_xlabel('Heterogeneity Level (α)', fontsize=13)
    ax.set_ylabel(f'{metric.upper()} MSE', fontsize=13)
    ax.set_title(f'{metric.upper()} MSE vs. Causal Heterogeneity', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, framealpha=0.9)
    ax.set_xticks(alphas)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_four_alpha_comparison(ate_results_by_alpha, cate_results_by_alpha, save_path=None, figsize=(14, 10)):
    """
    Create a 2x2 panel figure comparing Global vs CDV across 4 alpha values.
    
    Top row: scissors plots (ATE left, CATE right)
    Bottom row: bias-variance decomposition bars (ATE left, CATE right)
    
    Parameters:
    -----------
    ate_results_by_alpha : dict
        {alpha: {'global_mse_per_seed': [...], 'cdv_mse_per_seed': [...],
                 'global_bias_sq': float, 'global_variance': float,
                 'cdv_bias_sq': float, 'cdv_variance': float}}
    cate_results_by_alpha : dict
        Same structure as ate_results_by_alpha
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    alphas = sorted(ate_results_by_alpha.keys())
    
    # --- Top Left: ATE Scissors ---
    ax = axes[0, 0]
    g_means, c_means, g_cis, c_cis = [], [], [], []
    for alpha in alphas:
        r = ate_results_by_alpha[alpha]
        g_vals = np.array(r['global_mse_per_seed'])
        c_vals = np.array(r['cdv_mse_per_seed'])
        n = len(g_vals)
        g_means.append(np.mean(g_vals))
        c_means.append(np.mean(c_vals))
        g_cis.append(1.96 * np.std(g_vals) / np.sqrt(n))
        c_cis.append(1.96 * np.std(c_vals) / np.sqrt(n))
    
    ax.plot(alphas, g_means, 'o-', color='#d62728', linewidth=2, markersize=7, label='Global')
    ax.fill_between(alphas, np.array(g_means) - np.array(g_cis), np.array(g_means) + np.array(g_cis),
                    color='#d62728', alpha=0.15)
    ax.plot(alphas, c_means, 's-', color='#1f77b4', linewidth=2, markersize=7, label='CDV')
    ax.fill_between(alphas, np.array(c_means) - np.array(c_cis), np.array(c_means) + np.array(c_cis),
                    color='#1f77b4', alpha=0.15)
    ax.set_xlabel('α (heterogeneity)')
    ax.set_ylabel('ATE MSE')
    ax.set_title('ATE MSE vs. Heterogeneity', fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xticks(alphas)
    ax.grid(True, alpha=0.3)
    
    # --- Top Right: CATE Scissors ---
    ax = axes[0, 1]
    g_means, c_means, g_cis, c_cis = [], [], [], []
    for alpha in alphas:
        r = cate_results_by_alpha[alpha]
        g_vals = np.array(r['global_mse_per_seed'])
        c_vals = np.array(r['cdv_mse_per_seed'])
        n = len(g_vals)
        g_means.append(np.mean(g_vals))
        c_means.append(np.mean(c_vals))
        g_cis.append(1.96 * np.std(g_vals) / np.sqrt(n))
        c_cis.append(1.96 * np.std(c_vals) / np.sqrt(n))
    
    ax.plot(alphas, g_means, 'o-', color='#d62728', linewidth=2, markersize=7, label='Global')
    ax.fill_between(alphas, np.array(g_means) - np.array(g_cis), np.array(g_means) + np.array(g_cis),
                    color='#d62728', alpha=0.15)
    ax.plot(alphas, c_means, 's-', color='#1f77b4', linewidth=2, markersize=7, label='CDV')
    ax.fill_between(alphas, np.array(c_means) - np.array(c_cis), np.array(c_means) + np.array(c_cis),
                    color='#1f77b4', alpha=0.15)
    ax.set_xlabel('α (heterogeneity)')
    ax.set_ylabel('CATE MSE')
    ax.set_title('CATE MSE vs. Heterogeneity', fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xticks(alphas)
    ax.grid(True, alpha=0.3)
    
    # --- Bottom Left: ATE Bias-Variance Decomposition ---
    ax = axes[1, 0]
    x = np.arange(len(alphas))
    width = 0.35
    
    g_bias = [ate_results_by_alpha[a]['global_bias_sq'] for a in alphas]
    g_var = [ate_results_by_alpha[a]['global_variance'] for a in alphas]
    c_bias = [ate_results_by_alpha[a]['cdv_bias_sq'] for a in alphas]
    c_var = [ate_results_by_alpha[a]['cdv_variance'] for a in alphas]
    
    ax.bar(x - width/2, g_bias, width, label='Global Bias²', color='#d62728', alpha=0.8)
    ax.bar(x - width/2, g_var, width, bottom=g_bias, label='Global Variance', color='#d62728', alpha=0.4)
    ax.bar(x + width/2, c_bias, width, label='CDV Bias²', color='#1f77b4', alpha=0.8)
    ax.bar(x + width/2, c_var, width, bottom=c_bias, label='CDV Variance', color='#1f77b4', alpha=0.4)
    
    ax.set_xlabel('α (heterogeneity)')
    ax.set_ylabel('ATE MSE Components')
    ax.set_title('ATE Bias²-Variance Decomposition', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{a:.2f}' for a in alphas])
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    # --- Bottom Right: CATE Bias-Variance Decomposition ---
    ax = axes[1, 1]
    
    g_bias = [cate_results_by_alpha[a]['global_bias_sq'] for a in alphas]
    g_var = [cate_results_by_alpha[a]['global_variance'] for a in alphas]
    c_bias = [cate_results_by_alpha[a]['cdv_bias_sq'] for a in alphas]
    c_var = [cate_results_by_alpha[a]['cdv_variance'] for a in alphas]
    
    ax.bar(x - width/2, g_bias, width, label='Global Bias²', color='#d62728', alpha=0.8)
    ax.bar(x - width/2, g_var, width, bottom=g_bias, label='Global Variance', color='#d62728', alpha=0.4)
    ax.bar(x + width/2, c_bias, width, label='CDV Bias²', color='#1f77b4', alpha=0.8)
    ax.bar(x + width/2, c_var, width, bottom=c_bias, label='CDV Variance', color='#1f77b4', alpha=0.4)
    
    ax.set_xlabel('α (heterogeneity)')
    ax.set_ylabel('CATE MSE Components')
    ax.set_title('CATE Bias²-Variance Decomposition', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{a:.2f}' for a in alphas])
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    plt.show()
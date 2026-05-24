"""
Analysis Utilities for Sepsis Dataset

This module provides functions for analyzing variants and generating visualizations
for the sepsis dataset processing workflow.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def _hash(cols):
    """Generate a short hash from a list of column names."""
    import hashlib
    return hashlib.md5(",".join(sorted(cols)).encode()).hexdigest()[:8]


def analyze_variant_frequencies(pre_admission_vars):
    """
    Analyze the frequency distribution of pre-admission variable variants.
    
    Parameters:
    -----------
    pre_admission_vars : dict
        Dictionary mapping case_id to list of pre-admission variables
        
    Returns:
    --------
    tuple
        (variant_counts, sorted_variants, counts, cumulative_percentage, total_cases)
    """
    # Count how many cases have each unique combination of pre-admission variables
    variant_counts = Counter()
    for case_id, variables in pre_admission_vars.items():
        # Create a variant ID using hash function 
        variant_id = _hash(variables)
        variant_counts[variant_id] += 1

    # Sort variants by frequency (descending)
    sorted_variants = sorted(variant_counts.items(), key=lambda x: x[1], reverse=True)

    # Extract the counts only
    counts = [count for _, count in sorted_variants]

    # Calculate cumulative sum and percentage
    cumulative_counts = np.cumsum(counts)
    total_cases = sum(counts)
    cumulative_percentage = 100 * cumulative_counts / total_cases
    
    return variant_counts, sorted_variants, counts, cumulative_percentage, total_cases


def create_variant_elbow_chart(pre_admission_vars, figsize=(12, 6)):
    """
    Create an elbow chart showing variant frequency distribution.
    
    Parameters:
    -----------
    pre_admission_vars : dict
        Dictionary mapping case_id to list of pre-admission variables
    figsize : tuple
        Figure size for the plot
        
    Returns:
    --------
    None (displays plot)
    """
    variant_counts, sorted_variants, counts, cumulative_percentage, total_cases = analyze_variant_frequencies(pre_admission_vars)
    
    # Create elbow chart with dual y-axis
    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot the elbow chart (number of cases per variant)
    line1 = ax1.plot(range(1, len(counts) + 1), counts, marker='o', linestyle='-', color='blue', label='Cases per variant')
    ax1.set_xlabel('Variant Rank (sorted by frequency)')
    ax1.set_ylabel('Number of Cases', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create second y-axis for cumulative percentage
    ax2 = ax1.twinx()
    line2 = ax2.plot(range(1, len(counts) + 1), cumulative_percentage, marker='s', linestyle='--', color='red', label='Cumulative %')
    ax2.set_ylabel('Cumulative Percentage (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.grid(True)

    # Add a title
    plt.title('Elbow Chart: Number of Cases per Pre-admission Variables Variant')

    # Add a legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', bbox_to_anchor=(1, 0.8))

    # Add a text showing total number of variants
    plt.text(0.97, 0.1, f'Total variants: {len(counts)}', 
             transform=ax1.transAxes, ha='right', va='bottom', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add annotations for the top 5 variants
    for i in range(min(5, len(counts))):
        ax1.annotate(f'Variant {i+1}: {counts[i]} cases ({(counts[i]/total_cases*100):.1f}%)', 
                    xy=(i+1, counts[i]+3),
                    xytext=(15, 7), textcoords='offset points',
                    va='center', fontsize=8)

    plt.tight_layout()
    plt.show()

    # Display statistics
    print(f"Total number of variants: {len(counts)}")
    print(f"Most common variant has {counts[0]} cases ({counts[0]/total_cases*100:.2f}%)")
    
    if len(counts) >= 5:
        cumulative_counts = np.cumsum(counts)
        print(f"Top 5 variants cover {cumulative_counts[4]} cases out of {total_cases} total cases ({cumulative_percentage[4]:.2f}%)")
    
    variants_for_85_percent = np.argmax(cumulative_percentage >= 85) + 1 if any(cumulative_percentage >= 85) else len(counts)
    print(f"Number of variants needed to cover 85% of cases: {variants_for_85_percent}")


def analyze_admission_decision_distribution(df_sepsis):
    """
    Analyze the distribution of admission decisions (IC vs NC).
    
    Parameters:
    -----------
    df_sepsis : pd.DataFrame
        DataFrame with admission_decision column
        
    Returns:
    --------
    pd.Series
        Value counts with proportions
    """
    print("Admission Decision Distribution:")
    print("Counts:")
    print(df_sepsis['admission_decision'].value_counts(dropna=False))
    print("\nProportions:")
    proportions = df_sepsis['admission_decision'].value_counts(normalize=True, dropna=False)
    print(proportions)
    return proportions


def create_variant_admission_crosstab(df_sepsis_extended):
    """
    Create crosstabs showing the relationship between variants and admission decisions.
    
    Parameters:
    -----------
    df_sepsis_extended : pd.DataFrame
        DataFrame with variant and admission_decision columns
        
    Returns:
    --------
    tuple
        (absolute_crosstab, normalized_crosstab)
    """
    # Absolute counts
    absolute_crosstab = pd.crosstab(df_sepsis_extended['variant'], 
                                   df_sepsis_extended['admission_decision'], 
                                   margins=True)
    
    # Normalized by row (within each variant)
    normalized_crosstab = pd.crosstab(df_sepsis_extended['variant'], 
                                     df_sepsis_extended['admission_decision'], 
                                     margins=True, 
                                     normalize='index')
    
    print("Absolute counts:")
    print(absolute_crosstab)
    print("\nNormalized by variant (row percentages):")
    print(normalized_crosstab)
    
    return absolute_crosstab, normalized_crosstab


# =============================================================================
# SYNTHETIC DGP ANALYSIS UTILITIES
# =============================================================================


def create_synthetic_variant_elbow_chart(df, w_cols, figsize=(12, 6)):
    """
    Create an elbow chart for synthetic data showing feature-pattern frequency distribution.
    
    This mirrors create_variant_elbow_chart but works directly on a DataFrame
    with feature columns (using the binary pattern >=0 = present convention).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the feature columns
    w_cols : list
        List of feature column names
    figsize : tuple
        Figure size for the plot
        
    Returns:
    --------
    tuple
        (sorted_patterns, counts, cumulative_percentage, total_cases)
    """
    # Calculate binary feature patterns
    feature_mask = df[w_cols] >= 0
    patterns = feature_mask.apply(lambda row: ''.join(['1' if val else '0' for val in row]), axis=1)
    
    # Count pattern frequencies
    pattern_counts = Counter(patterns)
    
    # Sort by frequency (descending)
    sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
    pattern_names = [p for p, _ in sorted_patterns]
    counts = [c for _, c in sorted_patterns]
    
    # Calculate cumulative percentage
    cumulative_counts = np.cumsum(counts)
    total_cases = sum(counts)
    cumulative_percentage = 100 * cumulative_counts / total_cases
    
    # Create elbow chart with dual y-axis
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Plot the elbow chart (number of cases per variant)
    x_range = range(1, len(counts) + 1)
    line1 = ax1.plot(x_range, counts, marker='o', linestyle='-', color='blue', label='Cases per pattern')
    ax1.set_xlabel('Feature Pattern Rank (sorted by frequency)')
    ax1.set_ylabel('Number of Cases', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Create second y-axis for cumulative percentage
    ax2 = ax1.twinx()
    line2 = ax2.plot(x_range, cumulative_percentage, marker='s', linestyle='--', color='red', label='Cumulative %')
    ax2.set_ylabel('Cumulative Percentage (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.grid(True)
    
    # Add 85% coverage line
    ax2.axhline(y=85, color='green', linestyle=':', linewidth=2, alpha=0.7, label='85% coverage')
    
    # Title
    plt.title('Elbow Chart: Feature Pattern Frequency Distribution')
    
    # Legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    lines.append(plt.Line2D([0], [0], color='green', linestyle=':', linewidth=2))
    labels.append('85% coverage')
    ax1.legend(lines, labels, loc='center right')
    
    # Add text
    plt.text(0.97, 0.15, f'Total patterns: {len(counts)}', 
             transform=ax1.transAxes, ha='right', va='bottom', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Annotate top patterns with feature info
    for i in range(min(6, len(counts))):
        pattern = pattern_names[i]
        present_features = [w_cols[j] for j, bit in enumerate(pattern) if bit == '1']
        feature_str = ','.join(present_features)
        ax1.annotate(f'P{i+1}: [{feature_str}]\n{counts[i]} cases ({counts[i]/total_cases*100:.1f}%)', 
                    xy=(i+1, counts[i]),
                    xytext=(15, 10), textcoords='offset points',
                    va='center', fontsize=7,
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\nTotal unique feature patterns: {len(counts)}")
    print(f"Total cases: {total_cases}")
    print(f"\nTop patterns coverage:")
    for i in range(min(6, len(counts))):
        pattern = pattern_names[i]
        present_features = [w_cols[j] for j, bit in enumerate(pattern) if bit == '1']
        print(f"  Pattern {i+1} ({pattern}): features={present_features}, "
              f"n={counts[i]} ({counts[i]/total_cases*100:.1f}%), "
              f"cumulative={cumulative_percentage[i]:.1f}%")
    
    return sorted_patterns, counts, cumulative_percentage, total_cases
    
    return absolute_crosstab, normalized_crosstab
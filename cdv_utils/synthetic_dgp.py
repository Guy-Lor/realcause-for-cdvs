"""
Synthetic Data Generating Process (DGP) for CDV Evaluation

This module provides a fully controlled DGP with:
- 6 sub-groups (3 main variants covering >85% + 3 minor ones in "others" bucket)
- Per-variant causal structure (different features, propensity, treatment effects)
- A heterogeneity parameter alpha in [0, 1] that controls how different the variants are
- Known ground-truth CATE for every case

At alpha=0: all variants share the same treatment effect (homogeneous)
At alpha=1: variants have genuinely different causal mechanisms (heterogeneous)
"""

import numpy as np
import pandas as pd


# Sub-group definitions:
# Each sub-group specifies which features are present (non-negative) vs absent (-100)
# Feature columns: X1, X2, V, E, Z1, Z2
SUBGROUP_FEATURES = {
    0: ['X1', 'X2', 'V', 'Z1'],           # Variant 1: vitals + lab
    1: ['X1', 'X2', 'E', 'V'],            # Variant 2: ECG + vitals
    2: ['X1', 'X2', 'V', 'E', 'Z2'],      # Variant 3: vitals + ECG + special
    3: ['X1', 'X2', 'Z1', 'Z2'],           # Others: lab + special
    4: ['X1', 'X2', 'E', 'Z1'],            # Others: ECG + lab
    5: ['X1', 'X2', 'V', 'Z1', 'Z2'],      # Others: vitals + lab + special
}

# Population shares for 6 sub-groups (top 3 sum to 87%)
DEFAULT_VARIANT_SHARES = (0.40, 0.30, 0.17, 0.05, 0.04, 0.04)

# All feature columns in fixed order
ALL_W_COLS = ['X1', 'X2', 'V', 'E', 'Z1', 'Z2']

# Sentinel value for missing/absent features
MISSING_VALUE = -100.0

# Base treatment effect (same for all variants when alpha=0)
TAU_BASE = 5.0


def _sigmoid(x):
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _generate_features_for_subgroup(subgroup_id, X1, X2, rng):
    """
    Generate variant-specific features given shared features X1, X2.
    
    Each sub-group has different structural equations governing how features
    are generated, reflecting different causal structures.
    
    Returns dict of feature values (only for present features).
    """
    n = len(X1)
    features = {}
    
    if subgroup_id == 0:
        # Variant 1: V depends on X1+X2, Z1 depends on X1
        features['V'] = np.abs(rng.normal(X1 + 0.5 * X2, 1.0))
        features['Z1'] = np.abs(rng.normal(0.3 * X1 + 1.0, 0.8))
        
    elif subgroup_id == 1:
        # Variant 2: E depends on X1, V depends on X1 + 0.3*E (E causes V)
        features['E'] = np.abs(rng.normal(0.8 * X1 + 0.5, 1.0))
        features['V'] = np.abs(rng.normal(X1 + 0.3 * features['E'], 1.0))
        
    elif subgroup_id == 2:
        # Variant 3: V depends on X1+X2, E depends on X1 + 0.3*V (V causes E), Z2 depends on E
        features['V'] = np.abs(rng.normal(X1 + 0.5 * X2, 1.0))
        features['E'] = np.abs(rng.normal(0.5 * X1 + 0.3 * features['V'], 0.8))
        features['Z2'] = np.abs(rng.normal(0.4 * features['E'] + 0.5, 0.6))
        
    elif subgroup_id == 3:
        # Others group D: Z1 and Z2 independent of each other
        features['Z1'] = np.abs(rng.normal(0.5 * X1 + 1.0, 0.8))
        features['Z2'] = np.abs(rng.normal(0.3 * X2 + 1.2, 0.7))
        
    elif subgroup_id == 4:
        # Others group E: E and Z1
        features['E'] = np.abs(rng.normal(0.6 * X1 + 0.8, 1.0))
        features['Z1'] = np.abs(rng.normal(0.4 * X1 + 0.3 * features['E'], 0.8))
        
    elif subgroup_id == 5:
        # Others group F: V, Z1, Z2
        features['V'] = np.abs(rng.normal(X1 + 0.3 * X2, 1.0))
        features['Z1'] = np.abs(rng.normal(0.3 * features['V'] + 0.5, 0.7))
        features['Z2'] = np.abs(rng.normal(0.2 * X1 + 0.8, 0.6))
    
    return features


def _propensity_for_subgroup(subgroup_id, X1, X2, features):
    """
    Compute P(D=1 | features) for a given sub-group.
    Each sub-group has different confounding structure.
    """
    if subgroup_id == 0:
        logit = -0.5 + 0.4 * features['V'] - 0.3 * X1 + 0.2 * features['Z1']
    elif subgroup_id == 1:
        logit = -0.3 + 0.3 * features['E'] + 0.2 * features['V'] - 0.2 * X1
    elif subgroup_id == 2:
        logit = -0.4 + 0.3 * features['V'] + 0.2 * features['E'] - 0.15 * features['Z2']
    elif subgroup_id == 3:
        logit = -0.2 + 0.3 * features['Z1'] - 0.2 * features['Z2'] + 0.1 * X1
    elif subgroup_id == 4:
        logit = -0.3 + 0.35 * features['E'] + 0.2 * features['Z1'] - 0.15 * X1
    elif subgroup_id == 5:
        logit = -0.4 + 0.25 * features['V'] + 0.2 * features['Z1'] - 0.15 * features['Z2']
    
    return _sigmoid(logit)


def _baseline_outcome_for_subgroup(subgroup_id, X1, X2, features):
    """
    Compute baseline outcome Y(0) = f(features) + noise for a given sub-group.
    Different sub-groups have different outcome functions.
    """
    if subgroup_id == 0:
        baseline = 50.0 + 3.0 * X1 + 2.0 * features['V'] + 1.5 * features['Z1']
    elif subgroup_id == 1:
        baseline = 45.0 + 2.5 * X1 + 2.0 * features['E'] + 1.8 * features['V']
    elif subgroup_id == 2:
        baseline = 48.0 + 2.8 * X1 + 1.5 * features['V'] + 1.2 * features['E'] + features['Z2']
    elif subgroup_id == 3:
        baseline = 52.0 + 3.2 * X1 + 1.8 * features['Z1'] + 1.5 * features['Z2']
    elif subgroup_id == 4:
        baseline = 47.0 + 2.6 * X1 + 2.2 * features['E'] + 1.4 * features['Z1']
    elif subgroup_id == 5:
        baseline = 49.0 + 3.0 * X1 + 1.6 * features['V'] + 1.3 * features['Z1'] + features['Z2']
    
    return baseline


def _treatment_effect_for_subgroup(subgroup_id, alpha, X1, X2, features):
    """
    Compute heterogeneous treatment effect tau(features) for a given sub-group.
    
    tau = TAU_BASE + alpha * delta_v(features)
    
    At alpha=0: tau = TAU_BASE (constant for all, no heterogeneity)
    At alpha=1: tau varies by variant and features (full heterogeneity)
    
    Each sub-group uses a different nonlinear function delta_v.
    """
    if subgroup_id == 0:
        # delta_1: quadratic in V
        delta = 3.0 * (features['V'] - 2.0) ** 2 - 4.0 + 1.5 * features['Z1']
    elif subgroup_id == 1:
        # delta_2: interaction E * V
        delta = 2.0 * features['E'] * features['V'] / 3.0 - 5.0
    elif subgroup_id == 2:
        # delta_3: sinusoidal in V + linear in E
        delta = 4.0 * np.sin(features['V']) + 2.0 * features['E'] - 3.0 * features['Z2']
    elif subgroup_id == 3:
        # delta_4: product Z1*Z2
        delta = 2.5 * features['Z1'] * features['Z2'] / 2.0 - 3.0
    elif subgroup_id == 4:
        # delta_5: E squared minus Z1
        delta = 1.5 * features['E'] ** 2 / 3.0 - 2.0 * features['Z1']
    elif subgroup_id == 5:
        # delta_6: V + Z1 - Z2 interaction
        delta = 2.0 * features['V'] - 1.5 * features['Z1'] + features['Z2']
    
    return TAU_BASE + alpha * delta


def generate_synthetic_dataset(n, alpha, seed, variant_shares=None):
    """
    Generate a fully synthetic dataset with controlled causal heterogeneity.
    
    Parameters
    ----------
    n : int
        Number of cases to generate.
    alpha : float
        Heterogeneity parameter in [0, 1].
        0 = no heterogeneity (all variants same treatment effect)
        1 = full heterogeneity (variant-specific effects)
    seed : int
        Random seed for reproducibility.
    variant_shares : tuple of 6 floats, optional
        Population share for each of the 6 sub-groups (must sum to 1).
        Default: (0.40, 0.30, 0.17, 0.05, 0.04, 0.04)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: X1, X2, V, E, Z1, Z2, t, y, y0, y1, ite, subgroup
        Absent features are filled with MISSING_VALUE (-100).
    """
    if variant_shares is None:
        variant_shares = DEFAULT_VARIANT_SHARES
    
    assert len(variant_shares) == 6, "Must provide 6 sub-group shares"
    assert abs(sum(variant_shares) - 1.0) < 1e-6, "Shares must sum to 1"
    assert 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]"
    
    rng = np.random.default_rng(seed)
    
    # Step 1: Assign sub-groups
    subgroups = rng.choice(6, size=n, p=variant_shares)
    
    # Step 2: Generate shared features
    X1 = rng.uniform(0, 5, size=n)
    X2 = rng.binomial(1, 0.5, size=n).astype(float)
    
    # Step 3: Initialize feature arrays with sentinel
    V = np.full(n, MISSING_VALUE)
    E = np.full(n, MISSING_VALUE)
    Z1 = np.full(n, MISSING_VALUE)
    Z2 = np.full(n, MISSING_VALUE)
    
    # Step 4: Generate per-subgroup features, propensity, outcomes
    propensity = np.zeros(n)
    y0 = np.zeros(n)
    tau = np.zeros(n)
    
    for sg in range(6):
        mask = subgroups == sg
        n_sg = mask.sum()
        if n_sg == 0:
            continue
        
        X1_sg = X1[mask]
        X2_sg = X2[mask]
        
        # Generate features
        feat = _generate_features_for_subgroup(sg, X1_sg, X2_sg, rng)
        
        # Fill feature arrays
        if 'V' in feat:
            V[mask] = feat['V']
        if 'E' in feat:
            E[mask] = feat['E']
        if 'Z1' in feat:
            Z1[mask] = feat['Z1']
        if 'Z2' in feat:
            Z2[mask] = feat['Z2']
        
        # Propensity
        propensity[mask] = _propensity_for_subgroup(sg, X1_sg, X2_sg, feat)
        
        # Baseline outcome
        y0[mask] = _baseline_outcome_for_subgroup(sg, X1_sg, X2_sg, feat)
        
        # Treatment effect
        tau[mask] = _treatment_effect_for_subgroup(sg, alpha, X1_sg, X2_sg, feat)
    
    # Step 5: Add outcome noise
    noise_y0 = rng.normal(0, 2.0, size=n)
    y0 = y0 + noise_y0
    
    # Step 6: Draw treatment from propensity
    t = rng.binomial(1, propensity).astype(float)
    
    # Step 7: Compute potential outcomes and observed outcome
    y1 = y0 + tau
    ite = y1 - y0  # = tau
    y = y0 * (1 - t) + y1 * t
    
    # Build DataFrame
    df = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'V': V,
        'E': E,
        'Z1': Z1,
        'Z2': Z2,
        't': t,
        'y': y,
        'y0': y0,
        'y1': y1,
        'ite': ite,
        'subgroup': subgroups,
    })
    
    return df


def generate_counterfactuals_for_fixed_features(df_features, alpha, seed):
    """
    Given a DataFrame with fixed features (X1, X2, V, E, Z1, Z2, subgroup),
    regenerate treatment assignment and counterfactual outcomes for a specific alpha.
    
    This is used to keep test/val features constant while varying alpha.
    
    Parameters
    ----------
    df_features : pd.DataFrame
        DataFrame with columns X1, X2, V, E, Z1, Z2, subgroup
    alpha : float
        Heterogeneity parameter in [0, 1]
    seed : int
        Random seed
        
    Returns
    -------
    pd.DataFrame
        DataFrame with original features plus t, y, y0, y1, ite columns
    """
    rng = np.random.default_rng(seed)
    n = len(df_features)
    
    X1 = df_features['X1'].values
    X2 = df_features['X2'].values
    V = df_features['V'].values
    E = df_features['E'].values
    Z1 = df_features['Z1'].values
    Z2 = df_features['Z2'].values
    subgroups = df_features['subgroup'].values
    
    propensity = np.zeros(n)
    y0 = np.zeros(n)
    tau = np.zeros(n)
    
    for sg in range(6):
        mask = subgroups == sg
        n_sg = mask.sum()
        if n_sg == 0:
            continue
        
        X1_sg = X1[mask]
        X2_sg = X2[mask]
        
        # Reconstruct feature dict from stored values
        feat = {}
        present_features = SUBGROUP_FEATURES[sg]
        if 'V' in present_features:
            feat['V'] = V[mask]
        if 'E' in present_features:
            feat['E'] = E[mask]
        if 'Z1' in present_features:
            feat['Z1'] = Z1[mask]
        if 'Z2' in present_features:
            feat['Z2'] = Z2[mask]
        
        # Propensity
        propensity[mask] = _propensity_for_subgroup(sg, X1_sg, X2_sg, feat)
        
        # Baseline outcome
        y0[mask] = _baseline_outcome_for_subgroup(sg, X1_sg, X2_sg, feat)
        
        # Treatment effect
        tau[mask] = _treatment_effect_for_subgroup(sg, alpha, X1_sg, X2_sg, feat)
    
    # Outcome noise
    noise_y0 = rng.normal(0, 2.0, size=n)
    y0 = y0 + noise_y0
    
    # Treatment assignment
    t = rng.binomial(1, propensity).astype(float)
    
    # Potential outcomes
    y1 = y0 + tau
    ite = y1 - y0
    y = y0 * (1 - t) + y1 * t
    
    # Build result
    result = df_features.copy()
    result['t'] = t
    result['y'] = y
    result['y0'] = y0
    result['y1'] = y1
    result['ite'] = ite
    
    return result


def get_w_cols():
    """Return the list of feature column names."""
    return ALL_W_COLS.copy()


def get_ground_truth_variant_map():
    """
    Return the ground-truth mapping from modeled variant number to its feature set.
    For validation purposes only — not used in the modeling pipeline.
    """
    return {
        1: ['X1', 'X2', 'V', 'Z1'],
        2: ['X1', 'X2', 'E', 'V'],
        3: ['X1', 'X2', 'V', 'E', 'Z2'],
        4: 'all',  # others bucket uses all features
    }

"""
Causal Modeling Module

This module provides functions for causal inference analysis on variant-specific data,
including data preparation, model fitting, counterfactual prediction, and variant processing.
"""

import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def group_by_variants_with_filtered_columns(df, num_variants=3):
    """
    Group rows based on which features have non-NaN values and create separate dataframes
    for each variant with only the relevant columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data (should already have 'feature_pattern' and 'variant' columns)
    num_variants : int
        Number of variants to separate
        
    Returns:
    --------
    variant_dataframes : dict
        Dictionary with variant numbers as keys and filtered dataframes as values
    variant_info : pandas.DataFrame
        Information about each variant including size and pattern
    """
    # Identify feature columns (excluding outcome, treatment, and meta columns)
    feature_cols = [col for col in df.columns if col not in ['t', 'y', 'y0', 'y1', 'ite', 'feature_pattern', 'variant']]
    
    # Always include these essential columns
    essential_cols = ['t', 'y', 'y0', 'y1', 'ite']
    
    # Get unique patterns and their corresponding variant numbers
    pattern_variant_mapping = df[['feature_pattern', 'variant']].drop_duplicates().sort_values('variant')
    
    # Create variant info dataframe
    variant_info = []
    variant_dataframes = {}
    
    for _, row in pattern_variant_mapping.iterrows():
        pattern = row['feature_pattern']
        variant_num = row['variant']
        
        # Get rows for this variant
        variant_rows = df[df['variant'] == variant_num].copy()
        count = len(variant_rows)
        
        # Determine which feature columns to include based on pattern
        if variant_num == num_variants:
            # Last variant gets all columns (this handles the "others" group)
            relevant_feature_cols = feature_cols
        else:
            # For specific variants, only include columns where pattern has '1'
            relevant_feature_cols = [feature_cols[i] for i, bit in enumerate(pattern) if bit == '1']
        
        # Create the filtered dataframe with essential columns + relevant feature columns
        selected_columns = essential_cols + relevant_feature_cols
        variant_df = variant_rows[selected_columns].copy()
        
        # Store in dictionary
        variant_dataframes[variant_num] = variant_df
        
        # Create variant info
        feature_present = [pattern[i] == '1' for i in range(len(feature_cols))]
        feature_info = dict(zip(feature_cols, feature_present))
        
        variant_info.append({
            'variant': variant_num,
            'count': count,
            'percent': count / len(df) * 100,
            'pattern': pattern,
            'num_features': len(relevant_feature_cols),
            'feature_columns': relevant_feature_cols,
            **feature_info
        })
    
    variant_info_df = pd.DataFrame(variant_info)
    
    return variant_dataframes, variant_info_df


def assign_variants_by_patterns(df, top_variants, k):
    """
    Assign variants to data based on feature patterns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with feature data
    top_variants : list
        Most frequent variant patterns
    k : int
        Number of variants
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with assigned variants
    """
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in ['t', 'y', 'y0', 'y1', 'ite', 'variant','feature_pattern']]
    print(feature_cols)


    # Calculate feature patterns
    feature_mask = df[feature_cols] >= 0
    
    df = df.copy()
    df['feature_pattern'] = feature_mask.apply(lambda row: ''.join(['1' if val else '0' for val in row]), axis=1)
    
    # Assign variants
    def assign_variant(row):
        for i, variant in enumerate(top_variants, start=1):
            if row['feature_pattern'] == variant:
                return i
        return k  # Assign the k-th variant for all others
    
    df['variant'] = df.apply(assign_variant, axis=1)
    return df


def process_test_data_with_training_variants(test_df, training_variant_patterns, num_variants=3, for_global_model=False):
    """
    Process test data by assigning variants based on patterns learned from training data.
    
    Parameters:
    -----------
    test_df : pandas.DataFrame
        Test dataframe containing the data to be assigned to variants
    training_variant_patterns : dict
        Dictionary mapping variant numbers to their feature patterns from training data
    num_variants : int
        Number of variants (should match training)
    for_global_model : bool
        Whether processing for global model (affects feature column selection)
        
    Returns:
    --------
    test_variant_dataframes : dict
        Dictionary with variant numbers as keys and test dataframes as values
    test_variant_info : pandas.DataFrame
        Information about test variant assignments
    """
    # Get feature columns (same logic as training)
    feature_cols = [col for col in test_df.columns if col not in ['t', 'y', 'y0', 'y1', 'ite', 'feature_pattern', 'variant']]
    essential_cols = ['t', 'y', 'y0', 'y1', 'ite']
    
    # Calculate feature patterns for test data
    feature_mask = test_df[feature_cols] >= 0
    test_df = test_df.copy()
    test_df['feature_pattern'] = feature_mask.apply(lambda row: ''.join(['1' if val else '0' for val in row]), axis=1)
    
    # Create reverse mapping: pattern -> variant for training patterns
    pattern_to_variant = {pattern: variant_num for variant_num, pattern in training_variant_patterns.items()}
    
    # Assign variants based on training patterns
    def assign_test_variant(pattern):
        if pattern in pattern_to_variant:
            return pattern_to_variant[pattern]
        else:
            return num_variants  # Assign to last variant if no match
    
    test_df['variant'] = test_df['feature_pattern'].apply(assign_test_variant)
    
    # Create test variant dataframes
    test_variant_dataframes = {}
    test_variant_info = []
    
    for variant_num in range(1, num_variants + 1):
        # Get rows for this variant
        variant_rows = test_df[test_df['variant'] == variant_num].copy()
        count = len(variant_rows)
        
        if count == 0:
            print(f"Warning: No test cases found for Variant {variant_num}")
            continue
        
        # Determine feature columns based on training pattern
        if variant_num in training_variant_patterns and not for_global_model:
            # Use the pattern from training to determine relevant features
            training_pattern = training_variant_patterns[variant_num]
            relevant_feature_cols = [feature_cols[i] for i, bit in enumerate(training_pattern) if bit == '1']
        else:
            # For the "others" variant or global model, include all features
            relevant_feature_cols = feature_cols
        
        # Create the filtered dataframe
        selected_columns = essential_cols + relevant_feature_cols
        # Ensure all selected columns exist in the test data
        available_columns = [col for col in selected_columns if col in variant_rows.columns]
        test_variant_df = variant_rows[available_columns].copy()
        
        # Store in dictionary
        test_variant_dataframes[variant_num] = test_variant_df
        
        # Collect info
        test_variant_info.append({
            'variant': variant_num,
            'count': count,
            'percent': count / len(test_df) * 100,
            'num_features': len(relevant_feature_cols),
            'feature_columns': relevant_feature_cols,
            'is_others_bucket': variant_num == num_variants
        })
    
    test_variant_info_df = pd.DataFrame(test_variant_info)
    
    return test_variant_dataframes, test_variant_info_df


def setup_causal_estimators(seed=42):
    """
    Set up causal inference estimators for experiments.
    
    Parameters:
    -----------
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (estimators, estimators_for_global) - dictionaries of configured estimators
    """
    # Import here to avoid dependency issues
    from causal_estimators.metalearners import SLearner, TLearner, XLearner
    from causal_estimators.doubly_robust_estimator import DoublyRobustLearner
    from causal_estimators.double_ml import DoubleML
    
    # Base models
    rf_reg = RandomForestRegressor(n_estimators=50, random_state=seed)
    rf_clf = RandomForestClassifier(n_estimators=50, random_state=seed)
    ridge = Ridge(alpha=1.0, random_state=seed)
    
    estimators = {
        'S-Learner (Linear)': SLearner(outcome_model=LinearRegression()),
        'S-Learner (RF)': SLearner(outcome_model=rf_reg),
        'T-Learner (RF)': TLearner(outcome_models=rf_reg),
        'X-Learner (RF)': XLearner(
            outcome_models=rf_reg,
            cate_models=rf_reg,
            prop_score_model=rf_clf
        ),
        'DR Learner (EconML)': DoublyRobustLearner(
            outcome_model=rf_reg,
            prop_score_model=rf_clf,
            final_model=LinearRegression(),
            trim_eps=0.1,
            random_state=seed
        ),
        'Double ML': DoubleML(
            outcome_model=rf_reg,
            prop_score_model=rf_clf,
            final_model=LinearRegression(),
            discrete_treatment=True,
            random_state=seed
        ),
    }
    
    # Global estimators (can use different configurations if needed)
    estimators_for_global = {
        'S-Learner (Linear)': SLearner(outcome_model=LinearRegression()),
        'S-Learner (RF)': SLearner(outcome_model=rf_reg),
        'T-Learner (RF)': TLearner(outcome_models=rf_reg),
        'X-Learner (RF)': XLearner(
            outcome_models=rf_reg,
            cate_models=rf_reg,
            prop_score_model=rf_clf
        ),
        'DR Learner (EconML)': DoublyRobustLearner(
            outcome_model=rf_reg,
            prop_score_model=rf_clf,
            final_model=LinearRegression(),
            trim_eps=0.1,
            random_state=seed
        ),
        'Double ML': DoubleML(
            outcome_model=rf_reg,
            prop_score_model=rf_clf,
            final_model=LinearRegression(),
            discrete_treatment=True,
            random_state=seed
        ),
    }
    
    return estimators, estimators_for_global


def prepare_causal_data(variant_df):
    """
    Prepare data for causal modeling by extracting features, treatment, and outcome.
    
    Parameters:
    -----------
    variant_df : pandas.DataFrame
        Variant-specific dataframe
        
    Returns:
    --------
    tuple
        (X, t, y, feature_names) - features, treatment, outcome, feature column names
    """
    # Define columns to exclude
    exclude_cols = ['t', 'y', 'y0', 'y1', 'ite', 'feature_pattern', 'variant']
    
    # Get feature columns
    feature_cols = [col for col in variant_df.columns if col not in exclude_cols]
    
    # Extract data
    X = variant_df[feature_cols].values
    t = variant_df['t'].values
    y = variant_df['y'].values
    
    return X, t, y, feature_cols


def fit_estimator(estimator, X, t, y):
    """
    Fit a causal estimator to training data.
    
    Parameters:
    -----------
    estimator : causal estimator object
        The causal inference estimator to fit
    X : numpy.ndarray
        Feature matrix
    t : numpy.ndarray
        Treatment vector
    y : numpy.ndarray
        Outcome vector
        
    Returns:
    --------
    fitted_estimator : causal estimator object or None
        The fitted estimator, or None if fitting failed
    """
    try:
        # Ensure all arrays are 1D for t and y
        if hasattr(t, 'flatten'):
            t = t.flatten()
        if hasattr(y, 'flatten'):
            y = y.flatten()
        
        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Verify array lengths
        if len(t) != len(y) or len(t) != X.shape[0]:
            raise ValueError(f"Array length mismatch: X={X.shape[0]}, t={len(t)}, y={len(y)}")
        
        # Fit the estimator
        estimator.fit(X, t, y)
        
        return estimator
        
    except Exception as e:
        print(f"Error fitting estimator: {str(e)}")
        return None


def predict_counterfactuals(fitted_estimator, X, t, y):
    """
    Generate counterfactual predictions using a fitted estimator.
    
    Parameters:
    -----------
    fitted_estimator : causal estimator object
        The fitted causal inference estimator
    X : numpy.ndarray
        Feature matrix for prediction
    t : numpy.ndarray
        Treatment vector for prediction
    y : numpy.ndarray
        Outcome vector for prediction
        
    Returns:
    --------
    results_df : pandas.DataFrame or None
        DataFrame with observed and counterfactual predictions
    """
    try:
        # Ensure all arrays are 1D for t and y
        if hasattr(t, 'flatten'):
            t = t.flatten()
        if hasattr(y, 'flatten'):
            y = y.flatten()
        
        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Verify array lengths
        if len(t) != len(y) or len(t) != X.shape[0]:
            raise ValueError(f"Array length mismatch: X={X.shape[0]}, t={len(t)}, y={len(y)}")
        
        # Generate predictions for both treatment conditions
        y0_pred = fitted_estimator.predict_outcome(X, np.zeros_like(t))  # Control
        y1_pred = fitted_estimator.predict_outcome(X, np.ones_like(t))   # Treatment
        
        # Ensure predictions are 1D
        if hasattr(y0_pred, 'flatten'):
            y0_pred = y0_pred.flatten()
        if hasattr(y1_pred, 'flatten'):
            y1_pred = y1_pred.flatten()
        
        # Create results dataframe
        results_df = pd.DataFrame({
            't_observed': t,
            'y_observed': y,
            'y0_pred': y0_pred,
            'y1_pred': y1_pred,
        })
        
        # Calculate individual treatment effect
        results_df['ite_pred'] = results_df['y1_pred'] - results_df['y0_pred']
        
        return results_df
        
    except Exception as e:
        print(f"Error predicting counterfactuals: {str(e)}")
        return None


def fit_and_predict_all_estimators(variant_dataframes, test_variant_dataframes, 
                                  estimators, is_global=False):
    """
    Fit all estimators on training variants and predict on test variants.
    
    Parameters:
    -----------
    variant_dataframes : dict
        Training data by variant
    test_variant_dataframes : dict
        Test data by variant
    estimators : dict
        Dictionary of causal estimators
    is_global : bool
        Whether this is for global model fitting
        
    Returns:
    --------
    tuple
        (fitted_estimators, all_variant_results)
    """
    fitted_estimators = {}
    all_variant_results = {}
    
    for variant_num in sorted(variant_dataframes.keys()):
        # Get training data
        variant_df = variant_dataframes[variant_num]
        X_train, t_train, y_train, feature_cols = prepare_causal_data(variant_df)
        
        # Get test data
        if variant_num in test_variant_dataframes:
            test_variant_df = test_variant_dataframes[variant_num]
            X_test, t_test, y_test, _ = prepare_causal_data(test_variant_df)
        else:
            print(f"Warning: No test data for variant {variant_num}")
            continue
        
        fitted_estimators[variant_num] = {}
        all_variant_results[variant_num] = {}
        
        # Check if variant has sufficient data and treatment variation
        treatment_counts = variant_df['t'].value_counts()
        unique_treatments = variant_df['t'].unique()
        has_both_treatments = len(unique_treatments) >= 2 and all(treatment_counts >= 2)
        
        if not has_both_treatments and not is_global:
            print(f"Variant {variant_num}: Insufficient treatment variation, will use global model")
            continue
        
        # Fit estimators
        for estimator_name, estimator in estimators.items():
            fitted_estimator = fit_estimator(estimator, X_train, t_train, y_train)
            if fitted_estimator is not None:
                fitted_estimators[variant_num][estimator_name] = deepcopy(fitted_estimator)
                
                # Predict on test data
                results_df = predict_counterfactuals(fitted_estimator, X_test, t_test, y_test)
                if results_df is not None:
                    all_variant_results[variant_num][estimator_name] = results_df
    
    return fitted_estimators, all_variant_results


def select_best_model_per_variant(all_variant_val_results, val_variant_dataframes, 
                                 variants_to_skip=None, r2_threshold=0.2):
    """
    Select the best model per variant based on validation data.
    
    Parameters:
    -----------
    all_variant_val_results : dict
        Validation results by variant and estimator
    val_variant_dataframes : dict
        Validation data by variant
    variants_to_skip : list
        Variants that should use global model
    r2_threshold : float
        Minimum R² threshold for heterogeneity
        
    Returns:
    --------
    dict
        Best model selection info per variant
    """
    if variants_to_skip is None:
        variants_to_skip = []
    
    best_models_per_variant = {}
    
    for variant_num in sorted(all_variant_val_results.keys()):
        if variant_num in variants_to_skip:
            best_models_per_variant[variant_num] = {
                'estimator': "best_global_model",
                'ate_bias': None,
                'mse': None,
                'r2': None
            }
            continue
        
        # Get validation data for real ITE values
        val_df = val_variant_dataframes[variant_num]
        
        # Calculate metrics for each estimator
        estimator_scores = {}
        
        for estimator_name, results_df in all_variant_val_results[variant_num].items():
            ite_pred = results_df['ite_pred'].values
            ite_real = val_df['ite'].values
            
            # Calculate metrics
            ate_bias = abs(np.mean(ite_pred) - np.mean(ite_real))
            ite_mse = mean_squared_error(ite_real, ite_pred)
            ite_r2 = r2_score(ite_real, ite_pred)
            
            estimator_scores[estimator_name] = {
                'ate_bias': ate_bias,
                'mse': ite_mse,
                'r2': ite_r2
            }
        
        # Select best estimator using R² threshold + ATE bias minimization
        if estimator_scores:
            # Filter: keep models with R² > threshold
            valid_models = {k: v for k, v in estimator_scores.items() if v['r2'] > r2_threshold}
            
            if valid_models:
                # Among valid models, select lowest ATE bias
                best_estimator = min(valid_models.items(), key=lambda x: x[1]['ate_bias'])[0]
                best_metrics = valid_models[best_estimator]
            else:
                # Fallback: use lowest ATE bias anyway
                best_estimator = min(estimator_scores.items(), key=lambda x: x[1]['ate_bias'])[0]
                best_metrics = estimator_scores[best_estimator]
            
            best_models_per_variant[variant_num] = {
                'estimator': best_estimator,
                'ate_bias': best_metrics['ate_bias'],
                'mse': best_metrics['mse'],
                'r2': best_metrics['r2']
            }
        else:
            best_models_per_variant[variant_num] = {
                'estimator': "best_global_model",
                'ate_bias': None,
                'mse': None,
                'r2': None
            }
    
    return best_models_per_variant


def create_comprehensive_results_dataframe(all_variant_results, test_variant_dataframes, 
                                          method_name="variant", include_global_flag=True):
    """
    Create a comprehensive results dataframe from variant results.
    
    Parameters:
    -----------
    all_variant_results : dict
        Results by variant and estimator
    test_variant_dataframes : dict
        Test data by variant
    method_name : str
        Name for the method column
    include_global_flag : bool
        Whether to include used_global_model column
        
    Returns:
    --------
    pd.DataFrame
        Comprehensive results dataframe
    """
    predictions_list = []
    
    for variant_num in sorted(all_variant_results.keys()):
        test_df = test_variant_dataframes[variant_num]
        
        for estimator_name, results_df in all_variant_results[variant_num].items():
            temp_df = results_df.copy()
            temp_df['variant'] = variant_num
            temp_df['estimator'] = estimator_name
            temp_df['method'] = method_name
            
            if include_global_flag:
                temp_df['used_global_model'] = estimator_name == "best_global_model"
            
            # Add real counterfactuals
            temp_df['y0_real'] = test_df['y0'].values
            temp_df['y1_real'] = test_df['y1'].values
            temp_df['ite_real'] = test_df['ite'].values
            
            # Rename columns for clarity
            temp_df = temp_df.rename(columns={
                't_observed': 't',
                'y_observed': 'y',
                'y0_pred': 'y0_pred',
                'y1_pred': 'y1_pred',
                'ite_pred': 'ite_pred'
            })
            
            predictions_list.append(temp_df)
    
    if predictions_list:
        results_df = pd.concat(predictions_list, ignore_index=True)
        
        # Define column order
        if include_global_flag:
            column_order = ['variant', 'estimator', 'method', 'used_global_model', 't', 'y', 
                           'y0_real', 'y1_real', 'y0_pred', 'y1_pred', 'ite_real', 'ite_pred']
        else:
            column_order = ['variant', 'estimator', 'method', 't', 'y', 
                           'y0_real', 'y1_real', 'y0_pred', 'y1_pred', 'ite_real', 'ite_pred']
        
        # Ensure all columns exist before reordering
        available_columns = [col for col in column_order if col in results_df.columns]
        results_df = results_df[available_columns]
        
        return results_df
    else:
        return pd.DataFrame()
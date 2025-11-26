"""
Multi-Seed Experiment Runner Module

This module provides functions for running multi-seed experiments to assess
robustness and variability of causal inference predictions across different
random seeds and model configurations.
"""

import os
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error

from .causal_modeling import (
    assign_variants_by_patterns,
    group_by_variants_with_filtered_columns,
    process_test_data_with_training_variants,
    setup_causal_estimators,
    prepare_causal_data,
    fit_estimator,
    predict_counterfactuals,
    select_best_model_per_variant,
    create_comprehensive_results_dataframe
)

from .generator_validation import generate_synthetic_data, create_dataframe_from_synthetic_data


def run_single_seed_experiment(best_model, exp_seed, w_cols, top_variants, k, 
                              training_variant_patterns, test_variant_dataframes,
                              val_variant_dataframes, global_test_variant_dataframes,
                              global_val_variant_dataframes, initial_seed=420,
                              r2_threshold=0.2):
    """
    Run a single seed experiment for causal inference.
    
    Parameters:
    -----------
    best_model : TarNet
        Best trained RealCause model
    exp_seed : int
        Experiment seed
    w_cols : list
        Covariate column names
    top_variants : list
        Top variant patterns
    k : int
        Number of variants
    training_variant_patterns : dict
        Training variant patterns
    test_variant_dataframes : dict
        Test data by variant
    val_variant_dataframes : dict
        Validation data by variant  
    global_test_variant_dataframes : dict
        Global test data by variant
    global_val_variant_dataframes : dict
        Global validation data by variant
    initial_seed : int
        Initial random seed
    r2_threshold : float
        R² threshold for model selection
        
    Returns:
    --------
    dict
        Complete experiment results for this seed
    """
    print(f"[{exp_seed}] Step 1/5: Generating training data...")
    
    # ========================================================================
    # STEP 1: Generate new training data with current seed
    # ========================================================================
    w_train_exp, t_train_exp, y0_exp, y1_exp = generate_synthetic_data(
        best_model, seed=exp_seed, dataset='train'
    )
    
    # Create training dataframe
    w_samples_df_exp = create_dataframe_from_synthetic_data(
        w_train_exp, t_train_exp, y0_exp, y1_exp, w_cols
    )
    
    # Assign variants
    w_samples_df_exp = assign_variants_by_patterns(w_samples_df_exp, top_variants, k)
    
    # Create variant dataframes
    variant_dataframes_exp, _ = group_by_variants_with_filtered_columns(
        w_samples_df_exp, num_variants=k
    )
    
    # Create global dataframe
    w_samples_df_exp_global = w_samples_df_exp.copy()
    w_samples_df_exp_global['variant'] = 1
    global_variant_dataframes_exp, _ = group_by_variants_with_filtered_columns(
        w_samples_df_exp_global, num_variants=1
    )
    
    print(f"[{exp_seed}] Step 2/5: Initializing estimators...")
    
    # ========================================================================
    # STEP 2: Initialize estimators with current seed
    # ========================================================================
    estimators_exp, estimators_exp_global = setup_causal_estimators(seed=initial_seed)
    
    print(f"[{exp_seed}] Step 3/5: Training models and generating predictions...")
    
    # ========================================================================
    # STEP 3: Fit models and predict
    # ========================================================================
    
    # Determine variants to skip (use global model)
    max_variant_num_exp = max(variant_dataframes_exp.keys())
    variants_to_skip_exp = [max_variant_num_exp]  # Last variant always uses global
    
    for variant_num in sorted(variant_dataframes_exp.keys()):
        variant_df = variant_dataframes_exp[variant_num]
        treatment_counts = variant_df['t'].value_counts()
        unique_treatments = variant_df['t'].unique()
        has_both_treatments = len(unique_treatments) >= 2 and all(treatment_counts >= 2)
        
        if variant_num != max_variant_num_exp and not has_both_treatments:
            variants_to_skip_exp.append(variant_num)
    
    # FIT GLOBAL MODEL
    global_variant_df_exp = global_variant_dataframes_exp[1]
    X_global, t_global, y_global, _ = prepare_causal_data(global_variant_df_exp)
    
    # Storage for fitted models
    global_fitted_estimators_exp = {}
    for variant_num in range(1, len(training_variant_patterns) + 2):
        global_fitted_estimators_exp[variant_num] = {}
    
    # Fit global estimators
    for estimator_name, estimator in estimators_exp_global.items():
        fitted_estimator = fit_estimator(estimator, X_global, t_global, y_global)
        if fitted_estimator is not None:
            for variant_num in range(1, len(training_variant_patterns) + 2):
                global_fitted_estimators_exp[variant_num][estimator_name] = fitted_estimator
    
    # PREDICT WITH GLOBAL MODEL ON TEST DATA
    global_all_variant_results_exp = {}
    global_test_arrays_dict_exp = {}
    
    for variant_num in range(1, len(training_variant_patterns) + 2):
        global_test_variant_df_exp = global_test_variant_dataframes[variant_num]
        X_test_g, t_test_g, y_test_g, _ = prepare_causal_data(global_test_variant_df_exp)
        
        global_test_arrays_dict_exp[variant_num] = {
            'X_test': X_test_g.copy(),
            't_test': t_test_g.copy(),
            'y_test': y_test_g.copy()
        }
        
        global_all_variant_results_exp[variant_num] = {}
        
        for estimator_name, fitted_estimator in global_fitted_estimators_exp[variant_num].items():
            results_df = predict_counterfactuals(fitted_estimator, X_test_g, t_test_g, y_test_g)
            if results_df is not None:
                global_all_variant_results_exp[variant_num][estimator_name] = results_df
    
    # FIT VARIANT-SPECIFIC MODELS AND PREDICT ON TEST DATA
    all_variant_results_exp = {}
    fitted_estimators_exp = {}
    
    for variant_num in sorted(variant_dataframes_exp.keys()):
        if variant_num in variants_to_skip_exp:
            test_variant_df_exp = global_test_variant_dataframes[variant_num]
        else:
            test_variant_df_exp = test_variant_dataframes[variant_num]
        
        variant_df_exp = variant_dataframes_exp[variant_num]
        X_train, t_train, y_train, _ = prepare_causal_data(variant_df_exp)
        
        # Prepare test data
        if variant_num in variants_to_skip_exp:
            X_test_v = global_test_arrays_dict_exp[variant_num]['X_test']
            t_test_v = global_test_arrays_dict_exp[variant_num]['t_test']
            y_test_v = global_test_arrays_dict_exp[variant_num]['y_test']
        else:
            X_test_v, t_test_v, y_test_v, _ = prepare_causal_data(test_variant_df_exp)
        
        all_variant_results_exp[variant_num] = {}
        fitted_estimators_exp[variant_num] = {}
        
        # Fit estimators
        for estimator_name, estimator in estimators_exp.items():
            if variant_num in variants_to_skip_exp:
                fitted_estimators_exp[variant_num][estimator_name] = \
                    global_fitted_estimators_exp[variant_num][estimator_name]
            else:
                fitted_estimator = fit_estimator(estimator, X_train, t_train, y_train)
                if fitted_estimator is not None:
                    fitted_estimators_exp[variant_num][estimator_name] = deepcopy(fitted_estimator)
        
        # Predict on test data
        for estimator_name, fitted_estimator in fitted_estimators_exp[variant_num].items():
            results_df = predict_counterfactuals(fitted_estimator, X_test_v, t_test_v, y_test_v)
            if results_df is not None:
                all_variant_results_exp[variant_num][estimator_name] = results_df
    
    print(f"[{exp_seed}] Step 4/5: Predicting on validation data for best model selection...")
    
    # ========================================================================
    # STEP 4: Predict on VALIDATION data and select best model per variant
    # ========================================================================
    
    # Storage for validation predictions
    global_all_variant_val_results_exp = {}
    all_variant_val_results_exp = {}
    
    # PREDICT WITH GLOBAL MODEL ON VALIDATION DATA
    for variant_num in range(1, len(training_variant_patterns) + 2):
        global_val_variant_df_exp = global_val_variant_dataframes[variant_num]
        X_val_g, t_val_g, y_val_g, _ = prepare_causal_data(global_val_variant_df_exp)
        
        global_all_variant_val_results_exp[variant_num] = {}
        
        for estimator_name, fitted_estimator in global_fitted_estimators_exp[variant_num].items():
            results_df = predict_counterfactuals(fitted_estimator, X_val_g, t_val_g, y_val_g)
            if results_df is not None:
                global_all_variant_val_results_exp[variant_num][estimator_name] = results_df
    
    # PREDICT WITH VARIANT-SPECIFIC MODELS ON VALIDATION DATA
    for variant_num in sorted(variant_dataframes_exp.keys()):
        if variant_num in variants_to_skip_exp:
            val_variant_df_exp = global_val_variant_dataframes[variant_num]
        else:
            val_variant_df_exp = val_variant_dataframes[variant_num]
        
        X_val_v, t_val_v, y_val_v, _ = prepare_causal_data(val_variant_df_exp)
        
        all_variant_val_results_exp[variant_num] = {}
        
        # Predict
        for estimator_name, fitted_estimator in fitted_estimators_exp[variant_num].items():
            results_df = predict_counterfactuals(fitted_estimator, X_val_v, t_val_v, y_val_v)
            if results_df is not None:
                all_variant_val_results_exp[variant_num][estimator_name] = results_df
    
    # ========================================================================
    # SELECT BEST GLOBAL MODEL
    # ========================================================================
    print(f"[{exp_seed}] Selecting best global model...")
    
    # Combine validation data from all variants for global model evaluation
    global_val_combined_results = {}
    
    # Combine predictions from all variants for each global estimator
    for estimator_name in estimators_exp_global.keys():
        combined_predictions = []
        combined_real_ite = []
        
        for variant_num in global_all_variant_val_results_exp.keys():
            if estimator_name in global_all_variant_val_results_exp[variant_num]:
                variant_predictions = global_all_variant_val_results_exp[variant_num][estimator_name]['ite_pred'].values
                variant_real_ite = global_val_variant_dataframes[variant_num]['ite'].values
                
                combined_predictions.extend(variant_predictions)
                combined_real_ite.extend(variant_real_ite)
        
        if combined_predictions:
            global_val_combined_results[estimator_name] = {
                'ite_pred': np.array(combined_predictions),
                'ite_real': np.array(combined_real_ite)
            }
    
    # Calculate metrics for global model selection
    global_estimator_scores = {}
    
    for estimator_name, results in global_val_combined_results.items():
        ite_pred = results['ite_pred']
        ite_real = results['ite_real']
        
        # Calculate metrics
        ate_bias = abs(np.mean(ite_pred) - np.mean(ite_real))
        ite_mse = mean_squared_error(ite_real, ite_pred)
        ite_r2 = r2_score(ite_real, ite_pred)
        
        global_estimator_scores[estimator_name] = {
            'ate_bias': ate_bias,
            'mse': ite_mse,
            'r2': ite_r2
        }
    
    # Select best global model
    best_global_model_info = None
    best_global_model = None
    if global_estimator_scores:
        # Filter: keep models with R² > threshold
        valid_global_models = {k: v for k, v in global_estimator_scores.items() if v['r2'] > r2_threshold}
        
        if valid_global_models:
            best_global_estimator = min(valid_global_models.items(), key=lambda x: x[1]['ate_bias'])[0]
        else:
            best_global_estimator = min(global_estimator_scores.items(), key=lambda x: x[1]['ate_bias'])[0]
        
        best_global_model_info = {
            'estimator': best_global_estimator,
            'ate_bias': global_estimator_scores[best_global_estimator]['ate_bias'],
            'mse': global_estimator_scores[best_global_estimator]['mse'],
            'r2': global_estimator_scores[best_global_estimator]['r2']
        }
        
        best_global_model = global_fitted_estimators_exp[1][best_global_estimator]
    
    # ========================================================================
    # SELECT BEST MODEL PER VARIANT
    # ========================================================================
    print(f"[{exp_seed}] Selecting best models per variant...")
    
    best_models_per_variant = select_best_model_per_variant(
        all_variant_val_results_exp, val_variant_dataframes, 
        variants_to_skip_exp, r2_threshold
    )
    
    print(f"[{exp_seed}] Step 5/5: Creating result dataframes with best models...")
    
    # ========================================================================
    # STEP 5: Update variant method to use best models
    # ========================================================================
    
    # Update variant-specific results to use only the best model per variant
    updated_all_variant_results_exp = {}
    
    for variant_num in sorted(all_variant_results_exp.keys()):
        best_estimator_name = best_models_per_variant[variant_num]['estimator']
        updated_all_variant_results_exp[variant_num] = {}
        
        if best_estimator_name == "best_global_model":
            # Use the best global model
            X_test_v = global_test_arrays_dict_exp[variant_num]['X_test']
            t_test_v = global_test_arrays_dict_exp[variant_num]['t_test']
            y_test_v = global_test_arrays_dict_exp[variant_num]['y_test']
            
            if best_global_model is not None:
                results_df = predict_counterfactuals(best_global_model, X_test_v, t_test_v, y_test_v)
                if results_df is not None:
                    updated_all_variant_results_exp[variant_num]["best_global_model"] = results_df
        else:
            # Use the best variant-specific model
            if best_estimator_name in all_variant_results_exp[variant_num]:
                updated_all_variant_results_exp[variant_num][best_estimator_name] = \
                    all_variant_results_exp[variant_num][best_estimator_name]
    
    # ========================================================================
    # STEP 6: Create result dataframes
    # ========================================================================
    
    # Global method - all estimators
    global_method_all_exp = create_comprehensive_results_dataframe(
        global_all_variant_results_exp, global_test_variant_dataframes,
        method_name="global", include_global_flag=False
    )
    
    # Global method - best estimator only
    if best_global_model_info:
        best_global_estimator_name = best_global_model_info['estimator']
        global_method_best_exp = global_method_all_exp[
            global_method_all_exp['estimator'] == best_global_estimator_name
        ].copy()
    else:
        global_method_best_exp = pd.DataFrame()
    
    # Variant method - all estimators
    variant_method_all_exp = create_comprehensive_results_dataframe(
        all_variant_results_exp, test_variant_dataframes,
        method_name="variant", include_global_flag=True
    )
    
    # Variant method - best estimators only
    variant_method_best_exp = create_comprehensive_results_dataframe(
        updated_all_variant_results_exp, test_variant_dataframes,
        method_name="variant", include_global_flag=True
    )
    
    return {
        'global_method': global_method_all_exp,
        'global_method_best': global_method_best_exp,
        'variant_method': variant_method_all_exp,
        'variant_method_best': variant_method_best_exp,
        'best_model': best_models_per_variant,
        'best_global_model': best_global_model_info
    }


def run_multi_seed_experiment(best_model, experiment_seeds, w_cols, top_variants, k,
                             training_variant_patterns, test_variant_dataframes,
                             val_variant_dataframes, global_test_variant_dataframes,
                             global_val_variant_dataframes, results_save_path,
                             initial_seed=420, r2_threshold=0.2):
    """
    Run the complete multi-seed experiment.
    
    Parameters:
    -----------
    best_model : TarNet
        Best trained RealCause model
    experiment_seeds : list
        List of experiment seeds
    w_cols : list
        Covariate column names
    top_variants : list
        Top variant patterns
    k : int
        Number of variants
    training_variant_patterns : dict
        Training variant patterns
    test_variant_dataframes : dict
        Test data by variant
    val_variant_dataframes : dict
        Validation data by variant
    global_test_variant_dataframes : dict
        Global test data by variant
    global_val_variant_dataframes : dict
        Global validation data by variant
    results_save_path : str
        Path to save results
    initial_seed : int
        Initial random seed
    r2_threshold : float
        R² threshold for model selection
        
    Returns:
    --------
    dict
        Complete experiment results by seed
    """
    results_by_seed = {}
    
    # Create results directory if it doesn't exist
    # os.makedirs(results_save_path, exist_ok=True)
    
    print("=" * 70)
    print(f"Starting Multi-Seed Experiment with {len(experiment_seeds)} seeds")
    print("=" * 70)
    
    for seed_idx, exp_seed in enumerate(tqdm(experiment_seeds, desc="Processing seeds")):
        print(f"\n{'=' * 70}")
        print(f"SEED {exp_seed} ({seed_idx + 1}/{len(experiment_seeds)})")
        print(f"{'=' * 70}")
        
        try:
            seed_results = run_single_seed_experiment(
                best_model, exp_seed, w_cols, top_variants, k,
                training_variant_patterns, test_variant_dataframes,
                val_variant_dataframes, global_test_variant_dataframes,
                global_val_variant_dataframes, initial_seed, r2_threshold
            )
            
            results_by_seed[exp_seed] = seed_results
            
            # Save results after each seed
            with open(results_save_path, 'wb') as f:
                pickle.dump(results_by_seed, f)
            
            print(f"[{exp_seed}] ✓ Completed. Results saved to {results_save_path}")
            
        except Exception as e:
            print(f"[{exp_seed}] ✗ Failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 70)
    print("Multi-Seed Experiment Complete!")
    print("=" * 70)
    print(f"Total seeds processed: {len(results_by_seed)}")
    print(f"Results saved to: {results_save_path}")
    
    return results_by_seed


def load_experiment_results(results_save_path):
    """
    Load saved experiment results from pickle file.
    
    Parameters:
    -----------
    results_save_path : str
        Path to saved results file
        
    Returns:
    --------
    dict
        Loaded experiment results
    """
    try:
        with open(results_save_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Results file not found: {results_save_path}")
        return {}
    except Exception as e:
        print(f"Error loading results: {str(e)}")
        return {}


def analyze_experiment_results(results_by_seed):
    """
    Analyze the results from multi-seed experiment.
    
    Parameters:
    -----------
    results_by_seed : dict
        Results from multi-seed experiment
        
    Returns:
    --------
    dict
        Analysis summary
    """
    analysis = {
        'total_seeds': len(results_by_seed),
        'successful_seeds': [],
        'failed_seeds': [],
        'summary_by_seed': {}
    }
    
    for seed, results in results_by_seed.items():
        try:
            if results and 'global_method' in results:
                analysis['successful_seeds'].append(seed)
                analysis['summary_by_seed'][seed] = {
                    'global_method_shape': results['global_method'].shape,
                    'global_method_best_shape': results['global_method_best'].shape,
                    'variant_method_shape': results['variant_method'].shape,
                    'variant_method_best_shape': results['variant_method_best'].shape,
                    'best_global_model': results.get('best_global_model', None),
                    'best_models_count': len(results.get('best_model', {}))
                }
            else:
                analysis['failed_seeds'].append(seed)
        except Exception as e:
            analysis['failed_seeds'].append(seed)
    
    return analysis
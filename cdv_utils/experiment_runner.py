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

from .generator_validation import (
    generate_synthetic_data,
    create_dataframe_from_synthetic_data,
    estimate_sigmoid_flow_cate,
)


def _select_best_estimator_from_scores(estimator_scores, r2_threshold=0.2,
                                       selection_metric='ate_bias',
                                       use_r2_threshold=True):
    """Select the best estimator from validation scores by the requested metric."""
    supported_metrics = {'ate_bias', 'ate_mse', 'mse', 'pehe'}
    if selection_metric not in supported_metrics:
        raise ValueError(
            f"selection_metric must be one of {sorted(supported_metrics)}"
        )
    if not estimator_scores:
        return None

    valid_models = {
        name: score for name, score in estimator_scores.items()
        if score['r2'] > r2_threshold
    } if use_r2_threshold else estimator_scores
    candidate_scores = valid_models if valid_models else estimator_scores
    best_estimator = min(candidate_scores.items(), key=lambda x: x[1][selection_metric])[0]
    best_metrics = estimator_scores[best_estimator]

    return {
        'estimator': best_estimator,
        **best_metrics,
        'selection_metric': selection_metric,
    }


def run_single_seed_experiment(best_model, exp_seed, w_cols, top_variants, k, 
                              training_variant_patterns, test_variant_dataframes,
                              val_variant_dataframes, global_test_variant_dataframes,
                              global_val_variant_dataframes, initial_seed=420,
                              r2_threshold=0.2,
                              validation_cate_by_variant=None):
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
        Retained for API compatibility. The ATE-MSE and PEHE selectors use the
        requested validation error directly without an R² pre-filter.
    validation_cate_by_variant : dict, optional
        Corrected conditional-mean CATE arrays in the row order of each global
        validation dataframe. The outer multi-seed runner prepares this once.
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
        fitted_estimator = fit_estimator(estimator, X_global, t_global, y_global, estimator_name)
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
            results_df = predict_counterfactuals(fitted_estimator, X_test_g, t_test_g, y_test_g, estimator_name)
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
                fitted_estimator = fit_estimator(estimator, X_train, t_train, y_train, estimator_name)
                if fitted_estimator is not None:
                    fitted_estimators_exp[variant_num][estimator_name] = deepcopy(fitted_estimator)
        
        # Predict on test data
        for estimator_name, fitted_estimator in fitted_estimators_exp[variant_num].items():
            results_df = predict_counterfactuals(fitted_estimator, X_test_v, t_test_v, y_test_v, estimator_name)
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
            results_df = predict_counterfactuals(fitted_estimator, X_val_g, t_val_g, y_val_g, estimator_name)
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
            results_df = predict_counterfactuals(fitted_estimator, X_val_v, t_val_v, y_val_v, estimator_name)
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
                if validation_cate_by_variant is None:
                    raise ValueError(
                        "validation_cate_by_variant is required for corrected "
                        "ATE/CATE model selection."
                    )
                variant_real_ite = np.asarray(
                    validation_cate_by_variant[variant_num]
                ).reshape(-1)
                if len(variant_predictions) != len(variant_real_ite):
                    raise ValueError(
                        f"Variant {variant_num} has {len(variant_predictions)} "
                        f"validation predictions but {len(variant_real_ite)} "
                        "corrected CATE values."
                    )
                
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
            'ate_mse': ate_bias ** 2,
            'mse': ite_mse,
            'pehe': np.sqrt(ite_mse),
            'r2': ite_r2
        }
    
    # Select independently for aggregate-effect and heterogeneous-effect goals.
    # These are exact validation-error selectors, so no R² pre-filter is used.
    best_global_model_info = _select_best_estimator_from_scores(
        global_estimator_scores,
        r2_threshold=r2_threshold,
        selection_metric='ate_mse',
        use_r2_threshold=False,
    )
    best_global_cate_model_info = _select_best_estimator_from_scores(
        global_estimator_scores,
        r2_threshold=r2_threshold,
        selection_metric='pehe',
        use_r2_threshold=False,
    )
    best_global_model = None
    best_global_cate_model = None
    if best_global_model_info:
        best_global_model = global_fitted_estimators_exp[1][
            best_global_model_info['estimator']
        ]
    if best_global_cate_model_info:
        best_global_cate_model = global_fitted_estimators_exp[1][
            best_global_cate_model_info['estimator']
        ]
    
    # ========================================================================
    # SELECT BEST MODEL PER VARIANT
    # ========================================================================
    print(f"[{exp_seed}] Selecting best models per variant...")
    
    best_models_per_variant = select_best_model_per_variant(
        all_variant_val_results_exp,
        val_variant_dataframes,
        variants_to_skip_exp,
        r2_threshold,
        selection_metric='ate_mse',
        validation_effects_by_variant=validation_cate_by_variant,
        use_r2_threshold=False,
    )
    best_cate_models_per_variant = select_best_model_per_variant(
        all_variant_val_results_exp,
        val_variant_dataframes,
        variants_to_skip_exp,
        r2_threshold,
        selection_metric='pehe',
        validation_effects_by_variant=validation_cate_by_variant,
        use_r2_threshold=False,
    )
    
    print(f"[{exp_seed}] Step 5/5: Creating result dataframes with best models...")
    
    # ========================================================================
    # STEP 5: Update variant method to use best models
    # ========================================================================
    
    # Update variant-specific results to use only the best model per variant
    updated_all_variant_results_exp = {}
    updated_all_variant_cate_results_exp = {}
    
    for variant_num in sorted(all_variant_results_exp.keys()):
        best_estimator_name = best_models_per_variant[variant_num]['estimator']
        updated_all_variant_results_exp[variant_num] = {}
        
        if best_estimator_name == "best_global_model":
            # Use the best global model
            X_test_v = global_test_arrays_dict_exp[variant_num]['X_test']
            t_test_v = global_test_arrays_dict_exp[variant_num]['t_test']
            y_test_v = global_test_arrays_dict_exp[variant_num]['y_test']
            
            if best_global_model is not None:
                results_df = predict_counterfactuals(best_global_model, X_test_v, t_test_v, y_test_v, "best_global_model")
                if results_df is not None:
                    updated_all_variant_results_exp[variant_num]["best_global_model"] = results_df
        else:
            # Use the best variant-specific model
            if best_estimator_name in all_variant_results_exp[variant_num]:
                updated_all_variant_results_exp[variant_num][best_estimator_name] = \
                    all_variant_results_exp[variant_num][best_estimator_name]
    
        best_cate_estimator_name = best_cate_models_per_variant[variant_num]['estimator']
        updated_all_variant_cate_results_exp[variant_num] = {}

        if best_cate_estimator_name == "best_global_model":
            X_test_v = global_test_arrays_dict_exp[variant_num]['X_test']
            t_test_v = global_test_arrays_dict_exp[variant_num]['t_test']
            y_test_v = global_test_arrays_dict_exp[variant_num]['y_test']

            if best_global_cate_model is not None:
                results_df = predict_counterfactuals(
                    best_global_cate_model,
                    X_test_v,
                    t_test_v,
                    y_test_v,
                    "best_global_model",
                )
                if results_df is not None:
                    updated_all_variant_cate_results_exp[variant_num][
                        "best_global_model"
                    ] = results_df
        elif best_cate_estimator_name in all_variant_results_exp[variant_num]:
            updated_all_variant_cate_results_exp[variant_num][
                best_cate_estimator_name
            ] = all_variant_results_exp[variant_num][best_cate_estimator_name]

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
    
    if best_global_cate_model_info:
        best_global_cate_estimator_name = best_global_cate_model_info['estimator']
        global_method_best_cate_exp = global_method_all_exp[
            global_method_all_exp['estimator'] == best_global_cate_estimator_name
        ].copy()
    else:
        global_method_best_cate_exp = pd.DataFrame()

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
    
    variant_method_best_cate_exp = create_comprehensive_results_dataframe(
        updated_all_variant_cate_results_exp, test_variant_dataframes,
        method_name="variant", include_global_flag=True
    )

    return {
        'global_method': global_method_all_exp,
        'global_method_best': global_method_best_exp,
        'global_method_best_cate': global_method_best_cate_exp,
        'variant_method': variant_method_all_exp,
        'variant_method_best': variant_method_best_exp,
        'variant_method_best_cate': variant_method_best_cate_exp,
        'best_model': best_models_per_variant,
        'best_model_cate': best_cate_models_per_variant,
        'best_global_model': best_global_model_info,
        'best_global_model_cate': best_global_cate_model_info,
    }


def run_multi_seed_experiment(best_model, experiment_seeds, w_cols, top_variants, k,
                             training_variant_patterns, test_variant_dataframes,
                             val_variant_dataframes, global_test_variant_dataframes,
                             global_val_variant_dataframes, results_save_path,
                             initial_seed=420, r2_threshold=0.2,
                             validation_cate_by_variant=None,
                             cate_n_quantiles=128, cate_batch_rows=16):
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
        Retained for API compatibility. The ATE-MSE and PEHE selectors minimize
        their requested validation error without an R² pre-filter.
    validation_cate_by_variant : dict, optional
        Corrected CATE arrays aligned with the global validation dataframes.
        When omitted, they are computed once from ``best_model``.
    cate_n_quantiles : int
        Number of midpoint quantiles used for deterministic CATE quadrature.
    cate_batch_rows : int
        Number of validation rows integrated at a time.

    Returns:
    --------
    dict
        Complete experiment results by seed
    """
    results_by_seed = {}

    if validation_cate_by_variant is None:
        print(
            "Computing deterministic validation CATE for ATE-MSE and PEHE "
            "model selection..."
        )
        validation_cate_by_variant = {}
        for variant_num, validation_df in sorted(
            global_val_variant_dataframes.items()
        ):
            missing_features = [
                column for column in w_cols
                if column not in validation_df.columns
            ]
            if missing_features:
                raise ValueError(
                    f"Global validation variant {variant_num} is missing "
                    f"features required by RealCause: {missing_features}"
                )
            validation_cate_by_variant[variant_num] = estimate_sigmoid_flow_cate(
                best_model,
                validation_df[w_cols].to_numpy(),
                n_quantiles=cate_n_quantiles,
                batch_rows=cate_batch_rows,
            )

        print(
            "Corrected validation CATE prepared for variants: "
            f"{sorted(validation_cate_by_variant)}"
        )

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
                global_val_variant_dataframes, initial_seed, r2_threshold,
                validation_cate_by_variant
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


# =============================================================================
# SYNTHETIC DGP EXPERIMENT RUNNERS
# =============================================================================


def run_single_seed_synthetic_experiment(alpha, exp_seed, n_train, w_cols, top_variants, k,
                                         training_variant_patterns, test_variant_dataframes,
                                         val_variant_dataframes, global_test_variant_dataframes,
                                         global_val_variant_dataframes, initial_seed=420,
                                         r2_threshold=0.2, rf_n_jobs=-1):
    """
    Run a single seed experiment using the synthetic DGP instead of RealCause.
    
    Parameters:
    -----------
    alpha : float
        Heterogeneity parameter in [0, 1]
    exp_seed : int
        Experiment seed for data generation
    n_train : int
        Number of training samples to generate
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
        Initial random seed for estimator initialization
    r2_threshold : float
        R² threshold for model selection
    rf_n_jobs : int
        Number of parallel jobs for RandomForest models. Use -1 to use all
        available CPU cores.
        
    Returns:
    --------
    dict
        Complete experiment results for this seed
    """
    from .synthetic_dgp import generate_synthetic_dataset
    
    print(f"[{exp_seed}] Step 1/5: Generating synthetic training data (alpha={alpha})...")
    
    # ========================================================================
    # STEP 1: Generate new training data with current seed using synthetic DGP
    # ========================================================================
    df_train = generate_synthetic_dataset(n=n_train, alpha=alpha, seed=exp_seed)
    
    # Create training dataframe in the expected format (only w_cols + t, y, y0, y1, ite)
    w_samples_df_exp = df_train[w_cols + ['t', 'y', 'y0', 'y1', 'ite']].copy()
    
    # Assign variants using feature patterns
    w_samples_df_exp = assign_variants_by_patterns(w_samples_df_exp, top_variants, k)
    
    # Create variant dataframes (with filtered columns per variant)
    variant_dataframes_exp, _ = group_by_variants_with_filtered_columns(
        w_samples_df_exp, num_variants=k
    )
    
    # Create global dataframe (all data as one variant)
    w_samples_df_exp_global = w_samples_df_exp.copy()
    w_samples_df_exp_global['variant'] = 1
    w_samples_df_exp_global['feature_pattern'] = '1' * len(w_cols)
    global_variant_dataframes_exp, _ = group_by_variants_with_filtered_columns(
        w_samples_df_exp_global, num_variants=1
    )
    
    print(f"[{exp_seed}] Step 2/5: Initializing estimators...")
    
    # ========================================================================
    # STEP 2: Initialize estimators with current seed
    # ========================================================================
    estimators_exp, estimators_exp_global = setup_causal_estimators(
        seed=initial_seed, rf_n_jobs=rf_n_jobs
    )
    
    print(f"[{exp_seed}] Step 3/5: Training models and generating predictions...")
    
    # ========================================================================
    # STEP 3: Fit models and predict
    # ========================================================================
    
    # Determine variants to skip (use global model instead of variant-specific)
    # The "others" variant (last variant = k) always uses the global model,
    # since it pools heterogeneous leftover patterns with no single causal structure.
    # Any other variant with insufficient treatment/control also falls back to global.
    max_variant_num_exp = max(variant_dataframes_exp.keys())
    variants_to_skip_exp = [max_variant_num_exp]  # "others" always uses global
    
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
    for variant_num in range(1, k + 1):
        global_fitted_estimators_exp[variant_num] = {}
    
    # Fit global estimators
    for estimator_name, estimator in estimators_exp_global.items():
        fitted_estimator = fit_estimator(estimator, X_global, t_global, y_global, estimator_name)
        if fitted_estimator is not None:
            for variant_num in range(1, k + 1):
                global_fitted_estimators_exp[variant_num][estimator_name] = fitted_estimator
    
    # PREDICT WITH GLOBAL MODEL ON TEST DATA
    global_all_variant_results_exp = {}
    global_test_arrays_dict_exp = {}
    
    for variant_num in range(1, k + 1):
        if variant_num not in global_test_variant_dataframes:
            continue
        global_test_variant_df_exp = global_test_variant_dataframes[variant_num]
        X_test_g, t_test_g, y_test_g, _ = prepare_causal_data(global_test_variant_df_exp)
        
        global_test_arrays_dict_exp[variant_num] = {
            'X_test': X_test_g.copy(),
            't_test': t_test_g.copy(),
            'y_test': y_test_g.copy()
        }
        
        global_all_variant_results_exp[variant_num] = {}
        
        for estimator_name, fitted_estimator in global_fitted_estimators_exp[variant_num].items():
            results_df = predict_counterfactuals(fitted_estimator, X_test_g, t_test_g, y_test_g, estimator_name)
            if results_df is not None:
                global_all_variant_results_exp[variant_num][estimator_name] = results_df
    
    # FIT VARIANT-SPECIFIC MODELS AND PREDICT ON TEST DATA
    all_variant_results_exp = {}
    fitted_estimators_exp = {}
    
    for variant_num in sorted(variant_dataframes_exp.keys()):
        if variant_num in variants_to_skip_exp:
            test_variant_df_exp = global_test_variant_dataframes.get(variant_num)
        else:
            test_variant_df_exp = test_variant_dataframes.get(variant_num)
        
        if test_variant_df_exp is None:
            continue
        
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
                if estimator_name in global_fitted_estimators_exp.get(variant_num, {}):
                    fitted_estimators_exp[variant_num][estimator_name] = \
                        global_fitted_estimators_exp[variant_num][estimator_name]
            else:
                fitted_estimator = fit_estimator(estimator, X_train, t_train, y_train, estimator_name)
                if fitted_estimator is not None:
                    fitted_estimators_exp[variant_num][estimator_name] = deepcopy(fitted_estimator)
        
        # Predict on test data
        for estimator_name, fitted_estimator in fitted_estimators_exp[variant_num].items():
            results_df = predict_counterfactuals(fitted_estimator, X_test_v, t_test_v, y_test_v, estimator_name)
            if results_df is not None:
                all_variant_results_exp[variant_num][estimator_name] = results_df
    
    print(f"[{exp_seed}] Step 4/5: Predicting on validation data for best model selection...")
    
    # ========================================================================
    # STEP 4: Predict on VALIDATION data and select best model per variant
    # ========================================================================
    
    global_all_variant_val_results_exp = {}
    all_variant_val_results_exp = {}
    
    # PREDICT WITH GLOBAL MODEL ON VALIDATION DATA
    for variant_num in range(1, k + 1):
        if variant_num not in global_val_variant_dataframes:
            continue
        global_val_variant_df_exp = global_val_variant_dataframes[variant_num]
        X_val_g, t_val_g, y_val_g, _ = prepare_causal_data(global_val_variant_df_exp)
        
        global_all_variant_val_results_exp[variant_num] = {}
        
        for estimator_name, fitted_estimator in global_fitted_estimators_exp[variant_num].items():
            results_df = predict_counterfactuals(fitted_estimator, X_val_g, t_val_g, y_val_g, estimator_name)
            if results_df is not None:
                global_all_variant_val_results_exp[variant_num][estimator_name] = results_df
    
    # PREDICT WITH VARIANT-SPECIFIC MODELS ON VALIDATION DATA
    for variant_num in sorted(fitted_estimators_exp.keys()):
        if variant_num in variants_to_skip_exp:
            val_variant_df_exp = global_val_variant_dataframes.get(variant_num)
        else:
            val_variant_df_exp = val_variant_dataframes.get(variant_num)
        
        if val_variant_df_exp is None:
            continue
        
        X_val_v, t_val_v, y_val_v, _ = prepare_causal_data(val_variant_df_exp)
        
        all_variant_val_results_exp[variant_num] = {}
        
        for estimator_name, fitted_estimator in fitted_estimators_exp[variant_num].items():
            results_df = predict_counterfactuals(fitted_estimator, X_val_v, t_val_v, y_val_v, estimator_name)
            if results_df is not None:
                all_variant_val_results_exp[variant_num][estimator_name] = results_df
    
    # ========================================================================
    # SELECT BEST GLOBAL MODEL
    # ========================================================================
    print(f"[{exp_seed}] Selecting best global model...")
    
    global_val_combined_results = {}
    
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
    
    global_estimator_scores = {}
    
    for estimator_name, results in global_val_combined_results.items():
        ite_pred = results['ite_pred']
        ite_real = results['ite_real']
        
        ate_bias = abs(np.mean(ite_pred) - np.mean(ite_real))
        ite_mse = mean_squared_error(ite_real, ite_pred)
        ite_r2 = r2_score(ite_real, ite_pred)
        
        global_estimator_scores[estimator_name] = {
            'ate_bias': ate_bias,
            'mse': ite_mse,
            'r2': ite_r2
        }
    
    best_global_model_info = _select_best_estimator_from_scores(
        global_estimator_scores, r2_threshold=r2_threshold, selection_metric='ate_bias'
    )
    best_global_cate_model_info = _select_best_estimator_from_scores(
        global_estimator_scores, r2_threshold=r2_threshold, selection_metric='mse'
    )
    best_global_model = None
    best_global_cate_model = None
    if best_global_model_info:
        best_global_model = global_fitted_estimators_exp[1][best_global_model_info['estimator']]
    if best_global_cate_model_info:
        best_global_cate_model = global_fitted_estimators_exp[1][best_global_cate_model_info['estimator']]
    
    # ========================================================================
    # SELECT BEST MODEL PER VARIANT
    # ========================================================================
    print(f"[{exp_seed}] Selecting best models per variant...")
    
    best_models_per_variant = select_best_model_per_variant(
        all_variant_val_results_exp, val_variant_dataframes,
        variants_to_skip_exp, r2_threshold, selection_metric='ate_bias'
    )
    best_cate_models_per_variant = select_best_model_per_variant(
        all_variant_val_results_exp, val_variant_dataframes,
        variants_to_skip_exp, r2_threshold, selection_metric='mse'
    )
    
    print(f"[{exp_seed}] Step 5/5: Creating result dataframes with best models...")
    
    # ========================================================================
    # STEP 5: Update variant method to use best models
    # ========================================================================
    
    updated_all_variant_results_exp = {}
    updated_all_variant_cate_results_exp = {}
    
    for variant_num in sorted(all_variant_results_exp.keys()):
        best_estimator_name = best_models_per_variant[variant_num]['estimator']
        updated_all_variant_results_exp[variant_num] = {}
        
        if best_estimator_name == "best_global_model":
            X_test_v = global_test_arrays_dict_exp[variant_num]['X_test']
            t_test_v = global_test_arrays_dict_exp[variant_num]['t_test']
            y_test_v = global_test_arrays_dict_exp[variant_num]['y_test']
            
            if best_global_model is not None:
                results_df = predict_counterfactuals(best_global_model, X_test_v, t_test_v, y_test_v, "best_global_model")
                if results_df is not None:
                    updated_all_variant_results_exp[variant_num]["best_global_model"] = results_df
        else:
            if best_estimator_name in all_variant_results_exp[variant_num]:
                updated_all_variant_results_exp[variant_num][best_estimator_name] = \
                    all_variant_results_exp[variant_num][best_estimator_name]

        best_cate_estimator_name = best_cate_models_per_variant[variant_num]['estimator']
        updated_all_variant_cate_results_exp[variant_num] = {}

        if best_cate_estimator_name == "best_global_model":
            X_test_v = global_test_arrays_dict_exp[variant_num]['X_test']
            t_test_v = global_test_arrays_dict_exp[variant_num]['t_test']
            y_test_v = global_test_arrays_dict_exp[variant_num]['y_test']

            if best_global_cate_model is not None:
                results_df = predict_counterfactuals(
                    best_global_cate_model, X_test_v, t_test_v, y_test_v, "best_global_model"
                )
                if results_df is not None:
                    updated_all_variant_cate_results_exp[variant_num]["best_global_model"] = results_df
        else:
            if best_cate_estimator_name in all_variant_results_exp[variant_num]:
                updated_all_variant_cate_results_exp[variant_num][best_cate_estimator_name] = \
                    all_variant_results_exp[variant_num][best_cate_estimator_name]
    
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

    if best_global_cate_model_info:
        best_global_cate_estimator_name = best_global_cate_model_info['estimator']
        global_method_best_cate_exp = global_method_all_exp[
            global_method_all_exp['estimator'] == best_global_cate_estimator_name
        ].copy()
    else:
        global_method_best_cate_exp = pd.DataFrame()
    
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

    variant_method_best_cate_exp = create_comprehensive_results_dataframe(
        updated_all_variant_cate_results_exp, test_variant_dataframes,
        method_name="variant", include_global_flag=True
    )
    
    return {
        'global_method': global_method_all_exp,
        'global_method_best': global_method_best_exp,
        'global_method_best_cate': global_method_best_cate_exp,
        'variant_method': variant_method_all_exp,
        'variant_method_best': variant_method_best_exp,
        'variant_method_best_cate': variant_method_best_cate_exp,
        'best_model': best_models_per_variant,
        'best_model_cate': best_cate_models_per_variant,
        'best_global_model': best_global_model_info,
        'best_global_model_cate': best_global_cate_model_info
    }


def _normalize_alpha_values(alpha=None, alpha_values=None):
    if alpha_values is None:
        if alpha is None:
            raise ValueError("Either alpha or alpha_values must be provided.")
        alpha_values = [alpha]
    return list(alpha_values)


def _get_alpha_dataframes(dataframes, alpha, alpha_values, name):
    if len(alpha_values) == 1:
        if alpha in dataframes and isinstance(dataframes[alpha], dict):
            return dataframes[alpha]
        alpha_key = f"{alpha:.2f}"
        if alpha_key in dataframes and isinstance(dataframes[alpha_key], dict):
            return dataframes[alpha_key]
        return dataframes

    if alpha in dataframes:
        return dataframes[alpha]
    alpha_key = f"{alpha:.2f}"
    if alpha_key in dataframes:
        return dataframes[alpha_key]
    raise KeyError(f"{name} is missing dataframes for alpha={alpha}.")


def _get_alpha_results_save_path(results_save_path, alpha, alpha_values):
    if len(alpha_values) == 1:
        return results_save_path

    if "{alpha" in results_save_path:
        return results_save_path.format(alpha=alpha)

    root, ext = os.path.splitext(results_save_path)
    if ext:
        return f"{root}_alpha_{alpha:.2f}{ext}"
    return os.path.join(results_save_path, f"synthetic_alpha_{alpha:.2f}.pkl")


def _synthetic_result_has_required_keys(seed_result):
    required_keys = {
        'global_method',
        'global_method_best',
        'global_method_best_cate',
        'variant_method',
        'variant_method_best',
        'variant_method_best_cate',
        'best_model',
        'best_model_cate',
        'best_global_model',
        'best_global_model_cate',
    }
    return isinstance(seed_result, dict) and required_keys.issubset(seed_result.keys())


def run_multi_seed_synthetic_experiment(alpha=None, experiment_seeds=None, n_train=None, w_cols=None, top_variants=None, k=None,
                                        training_variant_patterns=None, test_variant_dataframes=None,
                                        val_variant_dataframes=None, global_test_variant_dataframes=None,
                                        global_val_variant_dataframes=None, results_save_path=None,
                                        initial_seed=420, r2_threshold=0.2, rf_n_jobs=-1,
                                        alpha_values=None):
    """
    Run the complete multi-seed synthetic experiment.

    Iteration order is seed-first: for each seed, run all requested alpha values.
    
    Parameters:
    -----------
    alpha : float, optional
        Single heterogeneity parameter in [0, 1]. Kept for backwards
        compatibility with existing single-alpha calls.
    alpha_values : list, optional
        Heterogeneity parameters in [0, 1]. When multiple alphas are supplied,
        test/validation dataframe arguments must be dictionaries keyed by alpha.
    experiment_seeds : list
        List of experiment seeds
    n_train : int
        Number of training samples to generate per seed
    w_cols : list
        Covariate column names
    top_variants : list
        Top variant patterns
    k : int
        Number of variants
    training_variant_patterns : dict
        Training variant patterns
    test_variant_dataframes : dict
        Test data by variant for single-alpha calls, or {alpha: data_by_variant}
        for multi-alpha calls.
    val_variant_dataframes : dict
        Validation data by variant for single-alpha calls, or {alpha: data_by_variant}
        for multi-alpha calls.
    global_test_variant_dataframes : dict
        Global test data by variant for single-alpha calls, or {alpha: data_by_variant}
        for multi-alpha calls.
    global_val_variant_dataframes : dict
        Global validation data by variant for single-alpha calls, or
        {alpha: data_by_variant} for multi-alpha calls.
    results_save_path : str
        Path to save results. For multi-alpha calls, use a format string such as
        "results/synthetic_alpha_{alpha:.2f}.pkl", a directory, or a base .pkl
        path that will be expanded with "_alpha_{alpha:.2f}".
    initial_seed : int
        Initial random seed
    r2_threshold : float
        R² threshold for model selection
    rf_n_jobs : int
        Number of parallel jobs for RandomForest models. Use -1 to use all
        available CPU cores.
        
    Returns:
    --------
    dict
        Complete experiment results as {alpha: {seed: results}}
    """
    alpha_values = _normalize_alpha_values(alpha=alpha, alpha_values=alpha_values)
    if experiment_seeds is None:
        raise ValueError("experiment_seeds must be provided.")
    if n_train is None:
        raise ValueError("n_train must be provided.")
    if results_save_path is None:
        raise ValueError("results_save_path must be provided.")

    results_by_alpha = {}
    save_paths_by_alpha = {
        alpha_value: _get_alpha_results_save_path(results_save_path, alpha_value, alpha_values)
        for alpha_value in alpha_values
    }

    for alpha_value, alpha_save_path in save_paths_by_alpha.items():
        if os.path.exists(alpha_save_path):
            with open(alpha_save_path, 'rb') as f:
                results_by_alpha[alpha_value] = pickle.load(f)
        else:
            results_by_alpha[alpha_value] = {}
    
    print("=" * 70)
    print(
        "Starting Synthetic Multi-Seed Experiment: "
        f"{len(experiment_seeds)} seeds x {len(alpha_values)} alphas"
    )
    print("=" * 70)
    
    for seed_idx, exp_seed in enumerate(tqdm(experiment_seeds, desc="Processing seeds")):
        print(f"\n{'=' * 70}")
        print(f"SEED {exp_seed} ({seed_idx + 1}/{len(experiment_seeds)})")
        print(f"{'=' * 70}")

        for alpha_idx, alpha_value in enumerate(alpha_values):
            print(f"\nALPHA {alpha_value} ({alpha_idx + 1}/{len(alpha_values)}) | seed={exp_seed}")

            if (
                exp_seed in results_by_alpha[alpha_value]
                and _synthetic_result_has_required_keys(results_by_alpha[alpha_value][exp_seed])
            ):
                print(f"[{exp_seed}] alpha={alpha_value} already completed. Skipping.")
                continue

            if exp_seed in results_by_alpha[alpha_value]:
                print(
                    f"[{exp_seed}] alpha={alpha_value} result is missing new CATE-selection keys. "
                    "Re-running this seed."
                )

            try:
                seed_results = run_single_seed_synthetic_experiment(
                    alpha=alpha_value, exp_seed=exp_seed, n_train=n_train,
                    w_cols=w_cols, top_variants=top_variants, k=k,
                    training_variant_patterns=training_variant_patterns,
                    test_variant_dataframes=_get_alpha_dataframes(
                        test_variant_dataframes, alpha_value, alpha_values, "test_variant_dataframes"
                    ),
                    val_variant_dataframes=_get_alpha_dataframes(
                        val_variant_dataframes, alpha_value, alpha_values, "val_variant_dataframes"
                    ),
                    global_test_variant_dataframes=_get_alpha_dataframes(
                        global_test_variant_dataframes, alpha_value, alpha_values, "global_test_variant_dataframes"
                    ),
                    global_val_variant_dataframes=_get_alpha_dataframes(
                        global_val_variant_dataframes, alpha_value, alpha_values, "global_val_variant_dataframes"
                    ),
                    initial_seed=initial_seed, r2_threshold=r2_threshold,
                    rf_n_jobs=rf_n_jobs
                )

                results_by_alpha[alpha_value][exp_seed] = seed_results

                alpha_save_path = save_paths_by_alpha[alpha_value]
                os.makedirs(os.path.dirname(alpha_save_path) or ".", exist_ok=True)
                with open(alpha_save_path, 'wb') as f:
                    pickle.dump(results_by_alpha[alpha_value], f)

                print(f"[{exp_seed}] alpha={alpha_value} completed. Results saved.")

            except Exception as e:
                print(f"[{exp_seed}] alpha={alpha_value} failed with error: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

    print("\n" + "=" * 70)
    print("Synthetic Experiment Complete!")
    print("=" * 70)
    for alpha_value in alpha_values:
        print(
            f"alpha={alpha_value}: {len(results_by_alpha[alpha_value])} seeds processed. "
            f"Results saved to: {save_paths_by_alpha[alpha_value]}"
        )

    if alpha is not None and alpha_values == [alpha]:
        return results_by_alpha[alpha]
    return results_by_alpha


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

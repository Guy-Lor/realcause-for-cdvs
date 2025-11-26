# """
# CDV Utils Package for Sepsis Dataset Analysis

# This package provides utilities for loading, processing, analyzing, and transforming
# the sepsis dataset for use with the RealCause framework.
# """
# try:
#     from .sepsis_loading import (
#         load_sepsis_data,
#         extract_admission_decisions,
#         extract_activity_sequences
#     )
# except ImportError:
#     print("Warning: pm4py is not installed. Sepsis data loading functions will be unavailable.")
#     load_sepsis_data = None
#     extract_admission_decisions = None
#     extract_activity_sequences = None

# from .feature_extraction import (
#     extract_pre_admission_variables_with_values,
#     extract_pre_admission_features,
#     assign_variants
# )

# from .analysis_utils import (
#     analyze_variant_frequencies,
#     create_variant_elbow_chart,
#     analyze_admission_decision_distribution,
#     create_variant_admission_crosstab
# )

# from .transform_utils import (
#     prepare_realcause_dataset,
#     encode_treatment_variable,
#     convert_boolean_columns,
#     filter_outliers,
#     clean_dataset,
#     save_realcause_dataset,
#     complete_transformation_pipeline
# )

# from .generator_validation import (
#     setup_model_architectures,
#     setup_training_parameters,
#     setup_outcome_distribution,
#     train_realcause_model,
#     train_multiple_models,
#     analyze_model_performance,
#     select_best_model,
#     validate_generator_quality,
#     generate_synthetic_data,
#     create_dataframe_from_synthetic_data
# )

# from .causal_modeling import (
#     group_by_variants_with_filtered_columns,
#     assign_variants_by_patterns,
#     process_test_data_with_training_variants,
#     setup_causal_estimators,
#     prepare_causal_data,
#     fit_estimator,
#     predict_counterfactuals,
#     fit_and_predict_all_estimators,
#     select_best_model_per_variant,
#     create_comprehensive_results_dataframe
# )

# from .experiment_runner import (
#     run_single_seed_experiment,
#     run_multi_seed_experiment,
#     load_experiment_results,
#     analyze_experiment_results
# )

# from .results_analysis import (
#     prepare_dataframes_for_analysis,
#     identify_fallback_variants,
#     add_error_columns,
#     calculate_ate_mse_decomposition,
#     calculate_ate_by_estimator,
#     calculate_ite_mse_pehe,
#     calculate_ite_mse_pehe_no_estimator,
#     perform_ate_mse_statistical_tests,
#     perform_cate_mse_statistical_tests,
#     create_comprehensive_statistical_table,
#     create_comprehensive_cate_statistical_table
# )

# from .visualization import (
#     setup_plotting_style,
#     plot_ate_bias_variance_tradeoff,
#     plot_cate_bias_variance_tradeoff,
#     plot_mse_comparison,
#     plot_statistical_results_summary,
#     create_results_visualization_dashboard
# )

# __all__ = [
#     # Loading utilities
#     'load_sepsis_data',
#     'extract_admission_decisions', 
#     'extract_activity_sequences',
    
#     # Feature extraction
#     'extract_pre_admission_variables_with_values',
#     'extract_pre_admission_features',
#     'assign_variants',
    
#     # Analysis utilities
#     'analyze_variant_frequencies',
#     'create_variant_elbow_chart',
#     'analyze_admission_decision_distribution',
#     'create_variant_admission_crosstab',
    
#     # Transform utilities
#     'prepare_realcause_dataset',
#     'encode_treatment_variable', 
#     'convert_boolean_columns',
#     'filter_outliers',
#     'clean_dataset',
#     'save_realcause_dataset',
#     'complete_transformation_pipeline',
    
#     # Generator validation
#     'setup_model_architectures',
#     'setup_training_parameters',
#     'setup_outcome_distribution',
#     'train_realcause_model',
#     'train_multiple_models',
#     'analyze_model_performance',
#     'select_best_model',
#     'validate_generator_quality',
#     'generate_synthetic_data',
#     'create_dataframe_from_synthetic_data',
    
#     # Causal modeling
#     'group_by_variants_with_filtered_columns',
#     'assign_variants_by_patterns',
#     'process_test_data_with_training_variants',
#     'setup_causal_estimators',
#     'prepare_causal_data',
#     'fit_estimator',
#     'predict_counterfactuals',
#     'fit_and_predict_all_estimators',
#     'select_best_model_per_variant',
#     'create_comprehensive_results_dataframe',
    
#     # Experiment runner
#     'run_single_seed_experiment',
#     'run_multi_seed_experiment',
#     'load_experiment_results',
#     'analyze_experiment_results',
    
#     # Results analysis
#     'prepare_dataframes_for_analysis',
#     'identify_fallback_variants',
#     'add_error_columns',
#     'calculate_ate_mse_decomposition',
#     'calculate_ate_by_estimator',
#     'calculate_ite_mse_pehe',
#     'calculate_ite_mse_pehe_no_estimator',
#     'perform_ate_mse_statistical_tests',
#     'perform_cate_mse_statistical_tests',
#     'create_comprehensive_statistical_table',
#     'create_comprehensive_cate_statistical_table',
    
#     # Visualization
#     'setup_plotting_style',
#     'plot_ate_bias_variance_tradeoff',
#     'plot_cate_bias_variance_tradeoff',
#     'plot_mse_comparison',
#     'plot_statistical_results_summary',
#     'create_results_visualization_dashboard'
# ]